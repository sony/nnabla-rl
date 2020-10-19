import nnabla as nn
import nnabla.functions as NF
import nnabla.solvers as NS

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
import nnabla_rl.models as M
import nnabla_rl.functions as RF
from nnabla_rl.distributions import Gaussian


def default_q_function_builder(scope_name, state_dim, action_dim):
    return M.TD3QFunction(scope_name, state_dim, action_dim)


def default_vae_builder(scope_name, state_dim, action_dim, latent_dim, max_action_value):
    return M.BCQVariationalAutoEncoder(scope_name, state_dim, action_dim, latent_dim, max_action_value)


def default_perturbator_builder(scope_name, state_dim, action_dim, max_action_value):
    return M.BCQPerturbator(scope_name, state_dim, action_dim, max_action_value)


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

    def __init__(self, env_info,
                 q_function_builder=default_q_function_builder,
                 vae_builder=default_vae_builder,
                 perturbator_builder=default_perturbator_builder,
                 params=BCQParam()):
        super(BCQ, self).__init__(env_info, params=params)

        state_dim = env_info.observation_space.shape[0]
        action_dim = env_info.action_space.shape[0]
        max_action_value = float(env_info.action_space.high[0])

        self._state_dim = state_dim
        self._action_dim = action_dim

        self._q_ensembles = []
        self._target_q_ensembles = []
        for i in range(self._params.num_q_ensembles):
            q = q_function_builder(
                scope_name="q{}".format(i), state_dim=state_dim, action_dim=action_dim)
            assert isinstance(q, M.QFunction)
            target_q = q_function_builder(
                scope_name="target_q{}".format(i), state_dim=state_dim, action_dim=action_dim)
            self._q_ensembles.append(q)
            self._target_q_ensembles.append(target_q)

        self._vae = vae_builder(scope_name="vae",
                                state_dim=state_dim,
                                action_dim=action_dim,
                                latent_dim=action_dim*2,
                                max_action_value=max_action_value)

        self._xi = perturbator_builder(scope_name="xi",
                                       state_dim=state_dim,
                                       action_dim=action_dim,
                                       max_action_value=max_action_value)
        self._target_xi = perturbator_builder(scope_name="target_xi",
                                              state_dim=state_dim,
                                              action_dim=action_dim,
                                              max_action_value=max_action_value)

        self._state = None
        self._action = None
        self._next_state = None
        self._replay_buffer = ReplayBuffer(capacity=None)

        # training input/loss variables
        self._s_current_var = nn.Variable((params.batch_size, state_dim))
        self._a_current_var = nn.Variable((params.batch_size, action_dim))
        self._s_next_var = nn.Variable((params.batch_size, state_dim))
        self._reward_var = nn.Variable((params.batch_size, 1))
        self._non_terminal_var = nn.Variable((params.batch_size, 1))
        self._vae_loss = None
        self._q_loss = None
        self._xi_loss = None

        latent_shape = (self._params.batch_size, action_dim * 2)
        self._target_latent_distribution = Gaussian(mean=np.zeros(shape=latent_shape, dtype=np.float32),
                                                    ln_var=np.zeros(shape=latent_shape, dtype=np.float32))

        # evaluation input/action variables
        self._eval_state_var = nn.Variable((1, state_dim))
        self._eval_action = None
        self._eval_max_index = None

    def _post_init(self):
        super(BCQ, self)._post_init()
        for q, target_q in zip(self._q_ensembles, self._target_q_ensembles):
            copy_network_parameters(
                q.get_parameters(), target_q.get_parameters(), 1.0)
        copy_network_parameters(self._xi.get_parameters(),
                                self._target_xi.get_parameters(),
                                1.0)

    def compute_eval_action(self, state):
        self._eval_state_var.d = np.expand_dims(state, axis=0)
        nn.forward_all([self._eval_action, self._eval_max_index])
        action = self._eval_action.d[self._eval_max_index.d[0]]
        return action

    def _build_training_graph(self):
        self._build_vae_update_graph()
        self._build_q_update_graph()
        self._build_xi_update_graph()

    def _build_evaluation_graph(self):
        repeat_num = 100
        state = RF.repeat(x=self._eval_state_var,
                          repeats=repeat_num,
                          axis=0)
        assert state.shape == (repeat_num, self._eval_state_var.shape[1])
        actions = self._vae.decode(state)
        noise = self._xi.generate_noise(state, actions, self._params.phi)
        self._eval_action = actions + noise
        q_values = self._q_ensembles[0].q(state, self._eval_action)
        self._eval_max_index = RF.argmax(q_values, axis=0)

    def _setup_solver(self):
        self._vae_solver = NS.Adam(alpha=self._params.learning_rate)
        self._vae_solver.set_parameters(self._vae.get_parameters())

        self._q_solvers = []
        for q in self._q_ensembles:
            solver = NS.Adam(alpha=self._params.learning_rate)
            solver.set_parameters(q.get_parameters())
            self._q_solvers.append(solver)

        self._xi_solver = NS.Adam(alpha=self._params.learning_rate)
        self._xi_solver.set_parameters(self._xi.get_parameters())

    def _run_online_training_iteration(self, env):
        raise NotImplementedError('BCQ does not support online training')

    def _run_offline_training_iteration(self, buffer):
        self._bcq_training(buffer)

    def _build_vae_update_graph(self):
        latent_distribution, reconstructed_action = self._vae(
            self._s_current_var, self._a_current_var)
        reconstruction_loss = RF.mean_squared_error(
            self._a_current_var, reconstructed_action)
        kl_divergence = \
            latent_distribution.kl_divergence(self._target_latent_distribution)
        latent_loss = 0.5 * NF.mean(kl_divergence)
        self._vae_loss = reconstruction_loss + latent_loss

    def _build_q_update_graph(self):
        s_next_rep = RF.repeat(
            x=self._s_next_var, repeats=self._params.num_action_samples, axis=0)
        assert s_next_rep.shape == (self._params.batch_size * self._params.num_action_samples,
                                    self._state_dim)
        a_next_rep = self._vae.decode(s_next_rep)
        assert a_next_rep.shape == (self._params.batch_size * self._params.num_action_samples,
                                    self._action_dim)
        noise = self._target_xi.generate_noise(
            s_next_rep, a_next_rep, phi=self._params.phi)
        q_values = NF.stack(*(q_target.q(s_next_rep, a_next_rep + noise)
                              for q_target in self._target_q_ensembles))
        assert q_values.shape == (self._params.num_q_ensembles,
                                  self._params.batch_size * self._params.num_action_samples,
                                  1)
        weighted_q_minmax = self._params.lmb * NF.min(q_values, axis=0) \
            + (1.0 - self._params.lmb) * NF.max(q_values, axis=0)
        assert weighted_q_minmax.shape == (
            self._params.batch_size * self._params.num_action_samples, 1)

        next_q_value = NF.max(
            NF.reshape(weighted_q_minmax, shape=(self._params.batch_size, -1)), axis=1, keepdims=True)
        assert next_q_value.shape == (self._params.batch_size, 1)
        target_q_value = self._reward_var + self._params.gamma * \
            self._non_terminal_var * next_q_value
        target_q_value.need_grad = False

        loss = 0.0
        for q in self._q_ensembles:
            loss += RF.mean_squared_error(target_q_value,
                                          q.q(self._s_current_var, self._a_current_var))
        self._q_loss = loss

    def _build_xi_update_graph(self):
        action = self._vae.decode(self._s_current_var)
        action.need_grad = False

        noise = self._xi.generate_noise(self._s_current_var,
                                        action,
                                        phi=self._params.phi)

        q_function = self._q_ensembles[0]
        xi_loss = -q_function.q(self._s_current_var, action + noise)
        assert xi_loss.shape == (self._params.batch_size, 1)

        self._xi_loss = NF.mean(xi_loss)

    def _bcq_training(self, replay_buffer):
        experiences, *_ = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next) = marshall_experiences(experiences)
        # Optimize critic
        self._s_current_var.d = s
        self._a_current_var.d = a
        self._s_next_var.d = s_next
        self._reward_var.d = r
        self._non_terminal_var.d = non_terminal

        # Train vae
        self._vae_loss.forward()
        self._vae_solver.zero_grad()
        self._vae_loss.backward()
        self._vae_solver.update()

        # Train q functions
        self._q_loss.forward()
        for q_solver in self._q_solvers:
            q_solver.zero_grad()
        self._q_loss.backward()
        for q_solver in self._q_solvers:
            q_solver.update()

        self._xi_loss.forward()
        self._xi_solver.zero_grad()
        self._xi_loss.backward()
        self._xi_solver.update()

        for q, target_q in zip(self._q_ensembles, self._target_q_ensembles):
            copy_network_parameters(
                q.get_parameters(), target_q.get_parameters(), self._params.tau)
        copy_network_parameters(self._xi.get_parameters(),
                                self._target_xi.get_parameters(),
                                self._params.tau)

    def _models(self):
        models = [*self._q_ensembles, *self._target_q_ensembles,
                  self._vae, self._xi, self._target_xi]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers['vae_solver'] = self._vae_solver
        for i, solver in enumerate(self._q_solvers):
            solvers['q{}_solver'.format(i)] = solver
        solvers['xi_solver'] = self._xi_solver
        return solvers

    @property
    def latest_iteration_state(self):
        state = super(BCQ, self).latest_iteration_state
        state['scalar']['vae_loss'] = self._vae_loss.d
        state['scalar']['q_loss'] = self._q_loss.d
        state['scalar']['xi_loss'] = self._xi_loss.d

        return state


if __name__ == "__main__":
    import nnabla_rl.environments as E
    env = E.DummyContinuous()
    bcq = BCQ(env)
