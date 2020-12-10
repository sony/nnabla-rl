import nnabla as nn
import nnabla.solvers as NS

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import TD3QFunction, BCQVariationalAutoEncoder, BCQPerturbator, QFunction, DeterministicPolicy
import nnabla_rl.model_trainers as MT
import nnabla_rl.functions as RF


def default_q_function_builder(scope_name, env_info, algorithm_params, **kwargs):
    return TD3QFunction(scope_name, env_info.state_dim, env_info.action_dim)


def default_vae_builder(scope_name, env_info, algorithm_params, **kwargs):
    max_action_value = float(env_info.action_space.high[0])
    return BCQVariationalAutoEncoder(scope_name,
                                     env_info.state_dim,
                                     env_info.action_dim,
                                     env_info.action_dim*2,
                                     max_action_value)


def default_perturbator_builder(scope_name, env_info, algorithm_params, **kwargs):
    max_action_value = float(env_info.action_space.high[0])
    return BCQPerturbator(scope_name,
                          env_info.state_dim,
                          env_info.action_dim,
                          max_action_value)


@dataclass
class BCQParam(AlgorithmParam):
    '''BCQParam
    Parameters used in BCQ algorithm.

    Args:
        tau(float): soft network parameter update coefficient. Defaults to 0.005.
        gamma(float): reward decay. Defaults to 0.99.
        learning_rate(float): learning rate which is set for solvers. Defaults to 2.0*1e-4.
        lmb(float): weight used for balancing the ratio of minQ and maxQ during q update. Defaults to 0.75.
        phi(float): action perturbator noise coefficient
        num_q_ensembles(int): number of q ensembles . Defaults to 2.
        num_action_sampoles(int): number of actions to sample for computing target q values. Defaults to 10.
        batch_size(int or None): Number of iterations starting to train the networks. Defaults to None.
    '''
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 1.0*1e-3
    lmb: float = 0.75
    phi: float = 0.05
    num_q_ensembles: int = 2
    num_action_samples: int = 10
    batch_size: int = 100

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        self._assert_between(self.tau, 0.0, 1.0, 'tau')
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.lmb, 'lmb')
        self._assert_positive(self.phi, 'phi')
        self._assert_positive(self.num_q_ensembles, 'num_q_ensembles')
        self._assert_positive(self.num_action_samples, 'num_action_samples')
        self._assert_positive(self.batch_size, 'batch_size')


class BCQ(Algorithm):
    '''Batch-Constrained Q-learning (BCQ) algorithm implementation.

    This class implements the Batch-Constrained Q-learning (BCQ) algorithm
    proposed by S. Fujimoto, et al. in the paper: "Off-Policy Deep Reinforcement Learning without Exploration"
    For detail see: https://arxiv.org/pdf/1812.02900.pdf

    '''

    def __init__(self, env_or_env_info,
                 q_function_builder=default_q_function_builder,
                 vae_builder=default_vae_builder,
                 perturbator_builder=default_perturbator_builder,
                 params=BCQParam()):
        super(BCQ, self).__init__(env_or_env_info, params=params)

        self._q_ensembles = []
        self._q_solvers = {}
        self._target_q_ensembles = []

        def solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        for i in range(self._params.num_q_ensembles):
            q = q_function_builder(scope_name=f"q{i}",
                                   env_info=self._env_info,
                                   algorithm_params=self._params)
            target_q = q.deepcopy(f'target_q{i}')
            assert isinstance(q, QFunction)
            assert isinstance(target_q, QFunction)
            self._q_ensembles.append(q)
            self._q_solvers[q.scope_name] = solver_builder()
            self._target_q_ensembles.append(target_q)

        self._vae = vae_builder(scope_name="vae", env_info=self._env_info, algorithm_params=self._params)
        self._vae_solver = {self._vae.scope_name: solver_builder()}

        self._xi = perturbator_builder(scope_name="xi", env_info=self._env_info, algorithm_params=self._params)
        self._xi_solver = {self._xi.scope_name: solver_builder()}
        self._target_xi = perturbator_builder(scope_name="target_xi",
                                              env_info=self._env_info,
                                              algorithm_params=self._params)

        self._replay_buffer = ReplayBuffer(capacity=None)

        self._q_function_trainer = None
        self._vae_trainer = None
        self._perturbator_trainer = None

    @eval_api
    def compute_eval_action(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            repeat_num = 100
            state = RF.repeat(x=self._eval_state_var, repeats=repeat_num, axis=0)
            assert state.shape == (repeat_num, self._eval_state_var.shape[1])
            actions = self._vae.decode(state)
            noise = self._xi.generate_noise(state, actions, self._params.phi)
            self._eval_action = actions + noise
            q_values = self._q_ensembles[0].q(state, self._eval_action)
            self._eval_max_index = RF.argmax(q_values, axis=0)
        self._eval_state_var.d = s
        nn.forward_all([self._eval_action, self._eval_max_index])
        return self._eval_action.d[self._eval_max_index.d[0]]

    def _before_training_start(self, env_or_buffer):
        self._vae_trainer = self._setup_vae_training(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._perturbator_trainer = self._setup_perturbator_training(env_or_buffer)

    def _setup_vae_training(self, env_or_buffer):
        trainer_params = MT.vae_trainers.KLDVariationalAutoEncoderTrainerParam()

        vae_trainer = MT.vae_trainers.KLDVariationalAutoEncoderTrainer(
            env_info=self._env_info,
            params=trainer_params)
        training = MT.model_trainer.Training()
        vae_trainer.setup_training(self._vae, self._vae_solver, training)
        return vae_trainer

    def _setup_q_function_training(self, env_or_buffer):
        trainer_params = MT.q_value_trainers.SquaredTDQFunctionTrainerParam(
            gamma=self._params.gamma,
            reduction_method='mean')

        q_function_trainer = MT.q_value_trainers.SquaredTDQFunctionTrainer(
            env_info=self._env_info,
            params=trainer_params)

        # This is a wrapper class which outputs the target action for next state in q function training
        class PerturbedPolicy(DeterministicPolicy):
            def __init__(self, vae, perturbator, phi):
                self._vae = vae
                self._perturbator = perturbator
                self._phi = phi

            def pi(self, s):
                a = self._vae.decode(s)
                noise = self._perturbator.generate_noise(s, a, phi=self._phi)
                return a + noise
        target_policy = PerturbedPolicy(self._vae, self._target_xi, self._params.phi)
        training = MT.q_value_trainings.BCQTraining(train_functions=self._q_ensembles,
                                                    target_functions=self._target_q_ensembles,
                                                    target_policy=target_policy,
                                                    num_action_samples=self._params.num_action_samples,
                                                    lmb=self._params.lmb)
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._q_ensembles,
            dst_models=self._target_q_ensembles,
            target_update_frequency=1,
            tau=self._params.tau)
        q_function_trainer.setup_training(self._q_ensembles, self._q_solvers, training)
        for q, target_q in zip(self._q_ensembles, self._target_q_ensembles):
            copy_network_parameters(q.get_parameters(), target_q.get_parameters(), 1.0)
        return q_function_trainer

    def _setup_perturbator_training(self, env_or_buffer):
        trainer_params = MT.perturbator_trainers.BCQPerturbatorTrainerParam(
            phi=self._params.phi
        )

        perturbator_trainer = MT.perturbator_trainers.BCQPerturbatorTrainer(
            env_info=self._env_info,
            params=trainer_params,
            q_function=self._q_ensembles[0],
            vae=self._vae)
        training = MT.model_trainer.Training()
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._xi,
            dst_models=self._target_xi,
            target_update_frequency=1,
            tau=self._params.tau)
        perturbator_trainer.setup_training(self._xi, self._xi_solver, training)
        copy_network_parameters(self._xi.get_parameters(), self._target_xi.get_parameters(), 1.0)
        return perturbator_trainer

    def _run_online_training_iteration(self, env):
        raise NotImplementedError('BCQ does not support online training')

    def _run_offline_training_iteration(self, buffer):
        self._bcq_training(buffer)

    def _bcq_training(self, replay_buffer):
        experiences, info = replay_buffer.sample(self._params.batch_size)
        marshalled_experiences = marshall_experiences(experiences)

        # Train vae
        self._vae_trainer.train(marshalled_experiences)

        kwargs = {}
        kwargs['weights'] = info['weights']
        errors = self._q_function_trainer.train(marshalled_experiences, **kwargs)
        td_error = np.abs(errors['td_error'])
        replay_buffer.update_priorities(td_error)

        self._perturbator_trainer.train(marshalled_experiences)

    def _models(self):
        models = [*self._q_ensembles, *self._target_q_ensembles,
                  self._vae, self._xi, self._target_xi]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers.update(self._vae_solver)
        solvers.update(self._q_solvers)
        solvers.update(self._xi_solver)
        return solvers


if __name__ == "__main__":
    import nnabla_rl.environments as E
    env = E.DummyContinuous()
    bcq = BCQ(env)
