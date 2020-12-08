from typing import Optional

import nnabla as nn
import nnabla.functions as NF
import nnabla.solvers as NS

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import TD3QFunction, BEARPolicy, UnsquashedVariationalAutoEncoder, \
    DeterministicPolicy, StochasticPolicy, QFunction, VariationalAutoEncoder
import nnabla_rl.model_trainers as MT
import nnabla_rl.functions as RF


def default_q_function_builder(scope_name, env_info, algorithm_params, **kwargs):
    return TD3QFunction(scope_name, env_info.state_dim, env_info.action_dim)


def default_policy_builder(scope_name, env_info, algorithm_params, **kwargs):
    return BEARPolicy(scope_name, env_info.state_dim, env_info.action_dim)


def default_vae_builder(scope_name, env_info, algorithm_params, **kwargs):
    return UnsquashedVariationalAutoEncoder(scope_name,
                                            env_info.state_dim,
                                            env_info.action_dim,
                                            env_info.action_dim*2)


@dataclass
class BEARParam(AlgorithmParam):
    '''BEARParam
    Parameters used in BEAR algorithm.

    Args:
        tau(float): soft network parameter update coefficient. Defaults to 0.005.
        gamma(float): reward decay. Defaults to 0.99.
        learning_rate(float): learning rate which is set for solvers. Defaults to 1.0*1e-3.
        lmb(float): weight used for balancing the ratio of minQ and maxQ during q update. Defaults to 0.75.
        epsilon(float): inequality constraint constant used during dual gradient descent. Defaults to 0.05.
        num_q_ensembles(int): number of q ensembles . Defaults to 2.
        num_mmd_actions(int): number of actions to sample for computing maximum mean discrepancy (MMD). Defaults to 5.
        num_action_sampoles(int): number of actions to sample for computing target q values. Defaults to 10.
        mmd_type(str): kernel type used for MMD computation. laplacian or gaussian is supported. Defaults to gaussian.
        mmd_sigma(float): parameter used for adjusting the  MMD. Defaults to 20.0.
        warmup_iterations(int): Number of iterations until start updating the policy. Defaults to 20000
        start_timesteps(int or None): Number of iterations to start training the networks.
                                      Only used on online training and must be set on online training.
                                      Defaults to None.
        batch_size(int or None): Number of iterations starting to train the networks. Defaults to None.
        use_mean_for_eval(bool): Use mean value instead of best action among the samples for evaluation
    '''
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 1e-3
    lmb: float = 0.75
    epsilon: float = 0.05
    num_q_ensembles: int = 2
    num_mmd_actions: int = 5
    num_action_samples: int = 10
    mmd_type: str = 'gaussian'
    mmd_sigma: float = 20.0
    initial_lagrange_multiplier: Optional[float] = None
    fix_lagrange_multiplier: bool = False
    warmup_iterations: int = 20000
    start_timesteps: int = None
    batch_size: int = 100
    use_mean_for_eval: bool = False

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        if not ((0.0 <= self.tau) & (self.tau <= 1.0)):
            raise ValueError('tau must lie between [0.0, 1.0]')
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')
        if not (0 <= self.num_q_ensembles):
            raise ValueError('num q ensembles must not be negative')
        if not (0 <= self.num_mmd_actions):
            raise ValueError('num mmd actions must not be negative')
        if not (0 <= self.num_action_samples):
            raise ValueError('num action samples must not be negative')
        if not (0 <= self.warmup_iterations):
            raise ValueError('warmup iterations must not be negative')
        if self.start_timesteps is not None:
            if not (0 <= self.start_timesteps):
                raise ValueError('start timesteps must not be negative')
        if not (0 <= self.batch_size):
            raise ValueError('batch size must not be negative')


class BEAR(Algorithm):
    '''Bootstrapping Error Accumulation Reduction (BEAR) algorithm implementation.

    This class implements the Bootstrapping Error Accumulation Reduction (BEAR) algorithm
    proposed by A. Kumar, et al. in the paper: "Stabilizing Off-Policy Q-learning via Bootstrapping Error Reduction"
    For detail see: https://arxiv.org/pdf/1906.00949.pdf

    '''

    def __init__(self, env_or_env_info,
                 q_function_builder=default_q_function_builder,
                 policy_builder=default_policy_builder,
                 vae_builder=default_vae_builder,
                 params=BEARParam()):
        super(BEAR, self).__init__(env_or_env_info, params=params)

        def solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._q_ensembles = []
        self._q_solvers = {}
        self._target_q_ensembles = []
        for i in range(self._params.num_q_ensembles):
            q = q_function_builder(
                scope_name="q{}".format(i), env_info=self._env_info, algorithm_params=self._params)
            assert isinstance(q, QFunction)
            target_q = q_function_builder(
                scope_name="target_q{}".format(i), env_info=self._env_info, algorithm_params=self._params)
            self._q_ensembles.append(q)
            self._q_solvers[q.scope_name] = solver_builder()
            self._target_q_ensembles.append(target_q)

        self._pi = policy_builder(scope_name="pi", env_info=self._env_info, algorithm_params=self._params)
        self._pi_solver = {self._pi.scope_name: solver_builder()}
        assert isinstance(self._pi, StochasticPolicy)
        self._target_pi = policy_builder(scope_name="target_pi", env_info=self._env_info, algorithm_params=self._params)
        assert isinstance(self._target_pi, StochasticPolicy)

        self._vae = vae_builder(scope_name="vae", env_info=self._env_info, algorithm_params=self._params)
        self._vae_solver = {self._vae.scope_name: solver_builder()}
        self._lagrange = MT.policy_trainers.bear_policy_trainer.AdjustableLagrangeMultiplier(
            scope_name="alpha",
            initial_value=self._params.initial_lagrange_multiplier)
        self._lagrange_solver = solver_builder()

        self._q_function_trainer = None
        self._vae_trainer = None
        self._policy_trainer = None

    def compute_eval_action(self, state):
        # evaluation input/action variables
        eval_state_var = nn.Variable((1, *state.shape))
        eval_state_var.d = np.expand_dims(state, axis=0)

        if self._params.use_mean_for_eval:
            with nn.auto_forward():
                eval_distribution = self._pi.pi(eval_state_var)
                eval_action = NF.tanh(eval_distribution.mean())
            return np.squeeze(eval_action.d, axis=0)
        else:
            with nn.auto_forward():
                repeat_num = 10
                state = RF.repeat(x=eval_state_var, repeats=repeat_num, axis=0)
                assert state.shape == (repeat_num, eval_state_var.shape[1])
                eval_distribution = self._pi.pi(state)
                eval_action = NF.tanh(eval_distribution.sample())
                q_values = self._q_ensembles[0].q(state, eval_action)
                eval_max_index = RF.argmax(q_values, axis=0)
            return eval_action.d[eval_max_index.d[0]]

    def _before_training_start(self, env_or_buffer):
        self._vae_trainer = self._setup_vae_training(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_vae_training(self, env_or_buffer):
        trainer_params = MT.vae_trainers.KLDVariationalAutoEncoderTrainerParam()

        vae_trainer = MT.vae_trainers.KLDVariationalAutoEncoderTrainer(
            env_info=self._env_info,
            params=trainer_params)

        # Wrapper for squashing reconstructed action during vae training
        class SquashedActionVAE(VariationalAutoEncoder):
            def __init__(self, original_vae):
                super().__init__(original_vae.scope_name)
                self._original_vae = original_vae

            def __call__(self, *args):
                latent_distribution, reconstructed = self._original_vae(*args)
                return latent_distribution, NF.tanh(reconstructed)

            def encode(self, *args): raise NotImplementedError
            def decode(self, *args): raise NotImplementedError
            def decode_multiple(self, decode_num, *args): raise NotImplementedError
            def latent_distribution(self, *args): raise NotImplementedError

        training = MT.model_trainer.Training()
        squashed_action_vae = SquashedActionVAE(self._vae)
        vae_trainer.setup_training(squashed_action_vae, self._vae_solver, training)
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
            def __init__(self, target_pi):
                super().__init__(target_pi.scope_name)
                self._target_pi = target_pi

            def pi(self, s):
                policy_distribution = self._target_pi.pi(s)
                return NF.tanh(policy_distribution.sample())
        target_policy = PerturbedPolicy(self._target_pi)
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

    def _setup_policy_training(self, env_or_buffer):
        trainer_params = MT.policy_trainers.BEARPolicyTrainerParam(
            num_mmd_actions=self._params.num_mmd_actions,
            mmd_type=self._params.mmd_type,
            epsilon=self._params.epsilon,
            fix_lagrange_multiplier=self._params.fix_lagrange_multiplier,
            warmup_iterations=self._params.warmup_iterations-self._iteration_num)

        class SquashedActionQ(QFunction):
            def __init__(self, original_q):
                super().__init__(original_q.scope_name)
                self._original_q = original_q

            def q(self, s, a):
                squashed_action = NF.tanh(a)
                return self._original_q.q(s, squashed_action)

        wrapped_qs = [SquashedActionQ(q) for q in self._q_ensembles]
        policy_trainer = MT.policy_trainers.BEARPolicyTrainer(
            self._env_info,
            q_ensembles=wrapped_qs,
            vae=self._vae,
            lagrange_multiplier=self._lagrange,
            lagrange_solver=self._lagrange_solver,
            params=trainer_params)
        training = MT.model_trainer.Training()
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._pi,
            dst_models=self._target_pi,
            target_update_frequency=1,
            tau=self._params.tau)
        policy_trainer.setup_training(self._pi, self._pi_solver, training)
        copy_network_parameters(self._pi.get_parameters(), self._target_pi.get_parameters(), 1.0)

        return policy_trainer

    def _run_online_training_iteration(self, env):
        raise NotImplementedError

    def _run_offline_training_iteration(self, buffer):
        self._bear_training(buffer)

    def _bear_training(self, replay_buffer):
        experiences, info = replay_buffer.sample(self._params.batch_size)
        marshalled_experiences = marshall_experiences(experiences)

        kwargs = {}
        kwargs['weights'] = info['weights']
        errors = self._q_function_trainer.train(marshalled_experiences, **kwargs)
        td_error = np.abs(errors['td_error'])
        replay_buffer.update_priorities(td_error)

        self._vae_trainer.train(marshalled_experiences)
        self._policy_trainer.train(marshalled_experiences)

    def _models(self):
        models = [*self._q_ensembles, *self._target_q_ensembles,
                  self._pi, self._target_pi, self._vae,
                  self._lagrange]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers.update(self._q_solvers)
        solvers.update(self._pi_solver)
        solvers.update(self._vae_solver)
        if not self._params.fix_lagrange_multiplier:
            solvers.update({self._lagrange.scope_name: self._lagrange_solver})
        return solvers

    @property
    def latest_iteration_state(self):
        state = super(BEAR, self).latest_iteration_state
        return state
