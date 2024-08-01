# Copyright 2023,2024 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass

import gym
import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.solvers as NS
import nnabla_rl.functions as RF
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms.common_utils import _ActionSelector
from nnabla_rl.algorithms.td3 import TD3, DefaultSolverBuilder, TD3Config
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer, EnvironmentExplorerConfig
from nnabla_rl.environment_explorers import RawPolicyExplorer, RawPolicyExplorerConfig
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import DeterministicPolicy, HyARPolicy, HyARQFunction, HyARVAE, QFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers import ReplacementSamplingReplayBuffer
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences, set_data_to_variable
from nnabla_rl.utils.misc import sync_model
from nnabla_rl.utils.solver_wrappers import AutoClipGradByNorm


@dataclass
class HyARConfig(TD3Config):
    """HyARConfig List of configurations for HyAR algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.003.
        batch_size(int): training batch size. Defaults to 100.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.\
            Defaults to 10000.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        d (int): Interval of the policy update. The policy will be updated every d q-function updates. Defaults to 2.
        exploration_noise_sigma (float): Standard deviation of the gaussian exploration noise. Defaults to 0.1.
        train_action_noise_sigma (float): Standard deviation of the gaussian action noise used in the training.\
            Defaults to 0.5.
        train_action_noise_abs (float): Absolute limit value of action noise used in the training. Defaults to 0.5.
        noisy_action_max (float): Maximum value of the training action after appending the noise. Defaults to 1.0.
        noisy_action_min (float): Minimum value of the training action after appending the noise. Defaults to -1.0.
        num_steps (int): number of steps for N-step Q targets. Defaults to 1.
        actor_unroll_steps (int): Number of steps to unroll actor's tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        actor_burn_in_steps (int): Number of burn-in steps to initiaze actor's recurrent layer states during training.\
            This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        actor_reset_rnn_on_terminal (bool): Reset actor's recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        critic_unroll_steps (int): Number of steps to unroll critic's tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        critic_burn_in_steps (int): Number of burn-in steps to initiaze critic's recurrent layer states\
            during training. This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        critic_reset_rnn_on_terminal (bool): Reset critic's recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        latent_dim (int): Latent action's dimension. Defaults to 6.\
        embed_dim (int): Discrete action embedding's dimension. Defaults to 6.\
        T (int): VAE training interval. VAE is trained every T episodes. Defaults to 10.\
        vae_pretrain_episodes (float): Number of data collection episodes for vae pretraining.\
            Defaults to 20000.\
        vae_pretrain_batch_size (int): Batch size used in vae pretraining.\
            Defaults to 64.\
        vae_pretrain_times (int): VAE is updated for this number of iterations during the pretrain stage.\
            Defaults to 5000.\
        vae_training_batch_size (int): batch size used in vae training. Defaults to 64.\
        vae_training_times (int): VAE is updated for this number of iterations every T steps. Defaults to 1.\
        vae_learning_rate (float): VAE learning rate. Defaults to 1e-4.\
        vae_buffer_size (int): Replay buffer size for VAE model. Defaults to 200000.\
        latent_select_batch_size: (int): Batch size for computing latent space constraint (LSC). Defaults to 5000.\
        latent_select_range: (float): Percentage of the latent variables in central range. Default to 96.\

        noise_decay_steps (int): Exploration noise decay steps. Noise decays for this number of experienced episodes.\
            Defaults to 1000.\
        initial_exploration_noise (float): Initial standard deviation of exploration noise. Defaults to 1.0.
        final_exploration_noise (float): Final standard deviation of exploration noise. Defaults to 0.1.
    """

    train_action_noise_sigma: float = 0.1
    train_action_noise_abs: float = 0.5
    noisy_action_min: float = -1.0
    noisy_action_max: float = -1.0

    latent_dim: int = 6
    embed_dim: int = 6
    T: int = 10
    vae_pretrain_episodes: int = 20000
    vae_pretrain_batch_size: int = 64
    vae_pretrain_times: int = 5000
    vae_training_batch_size: int = 64
    vae_training_times: int = 1
    vae_learning_rate: float = 1e-4
    vae_buffer_size: int = int(2e6)

    latent_select_batch_size: int = 5000
    latent_select_range: float = 96.0

    noise_decay_steps: int = 1000
    initial_exploration_noise: float = 1.0
    final_exploration_noise: float = 0.1

    def __post_init__(self):
        self._assert_positive(self.latent_dim, "latent_dim")
        self._assert_positive(self.embed_dim, "embed_dim")
        self._assert_positive(self.T, "T")
        self._assert_positive_or_zero(self.vae_pretrain_episodes, "vae_pretrain_episodes")
        self._assert_positive(self.vae_pretrain_batch_size, "vae_pretrain_batch_size")
        self._assert_positive_or_zero(self.vae_pretrain_times, "vae_pretrain_times")
        self._assert_positive(self.vae_training_batch_size, "vae_training_batch_size")
        self._assert_positive_or_zero(self.vae_training_times, "vae_training_times")
        self._assert_positive_or_zero(self.vae_learning_rate, "vae_learning_rate")
        self._assert_positive(self.vae_buffer_size, "vae_buffer_size")
        self._assert_positive(self.latent_select_batch_size, "latent_select_batch_size")
        self._assert_between(self.latent_select_range, 0, 100, "latent_select_range")
        self._assert_positive(self.noise_decay_steps, "noise_decay_steps")
        self._assert_positive(self.initial_exploration_noise, "initial_exploration_noise")
        self._assert_positive(self.final_exploration_noise, "final_exploration_noise")
        return super().__post_init__()


class DefaultCriticBuilder(ModelBuilder[QFunction]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: HyARConfig,
        **kwargs,
    ) -> QFunction:
        return HyARQFunction(scope_name)


class DefaultActorBuilder(ModelBuilder[DeterministicPolicy]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: HyARConfig,
        **kwargs,
    ) -> DeterministicPolicy:
        max_action_value = 1.0
        action_dim = algorithm_config.latent_dim + algorithm_config.embed_dim
        return HyARPolicy(scope_name, action_dim, max_action_value=max_action_value)


class DefaultVAEBuilder(ModelBuilder[HyARVAE]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: HyARConfig,
        **kwargs,
    ) -> HyARVAE:
        return HyARVAE(
            scope_name,
            state_dim=env_info.state_dim,
            action_dim=env_info.action_dim,
            encode_dim=algorithm_config.latent_dim,
            embed_dim=algorithm_config.embed_dim,
        )


class DefaultActorSolverBuilder(SolverBuilder):
    def build_solver(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: HyARConfig, **kwargs
    ) -> nn.solver.Solver:
        solver = NS.Adam(alpha=algorithm_config.learning_rate)
        return AutoClipGradByNorm(solver, 10.0)


class DefaultVAESolverBuilder(SolverBuilder):
    def build_solver(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: HyARConfig, **kwargs
    ) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.vae_learning_rate)


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(  # type: ignore[override]
        self,
        env_info: EnvironmentInfo,
        algorithm_config: HyARConfig,
        algorithm: "HyAR",
        **kwargs,
    ) -> EnvironmentExplorer:
        explorer_config = HyARPolicyExplorerConfig(
            warmup_random_steps=0, initial_step_num=algorithm.iteration_num, timelimit_as_terminal=False
        )
        explorer = HyARPolicyExplorer(
            policy_action_selector=algorithm._exploration_action_selector, env_info=env_info, config=explorer_config
        )
        return explorer


class DefaultPretrainExplorerBuilder(ExplorerBuilder):
    def build_explorer(  # type: ignore[override]
        self,
        env_info: EnvironmentInfo,
        algorithm_config: HyARConfig,
        algorithm: "HyAR",
        **kwargs,
    ) -> EnvironmentExplorer:
        explorer_config = HyARPretrainExplorerConfig(
            warmup_random_steps=0, initial_step_num=algorithm.iteration_num, timelimit_as_terminal=False
        )
        explorer = HyARPretrainExplorer(env_info=env_info, config=explorer_config)
        return explorer


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: HyARConfig, **kwargs
    ) -> ReplayBuffer:
        return ReplacementSamplingReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class DefaultVAEBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: HyARConfig, **kwargs
    ) -> ReplayBuffer:
        return ReplacementSamplingReplayBuffer(capacity=algorithm_config.vae_buffer_size)


class HyAR(TD3):
    """HyAR algorithm.

    This class implements the Hybrid Action Representation (HyAR) algorithm
    proposed by Boyan Li, et al.
    in the paper: "HyAR: Addressing Discrete-Continuous Action Reinforcement Learning via Hybrid Action Representation"
    For details see: https://openreview.net/pdf?id=64trBbOhdGU

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`DQNConfig <nnabla_rl.algorithms.dqn.DQNConfig>`):
            the parameter for DQN training
        critic_func_builder (:py:class:`ModelBuilder <nnabla_rl.builders.ModelBuilder>`): builder of q function model
        critic_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of q function solver
        actor_func_builder (:py:class:`ModelBuilder <nnabla_rl.builders.ModelBuilder>`): builder of policy model
        actor_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder of policy solver
        vae_builder (:py:class:`ModelBuilder <nnabla_rl.builders.ModelBuilder>`): builder of vae model
        vae_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder of vae solver
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of q-function and policy replay_buffer
        vae_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of vae's replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer for main training stage
        pretrain_explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer for pretraining stage
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: HyARConfig
    _evaluation_actor: "_HyARPolicyActionSelector"  # type: ignore
    _exploration_actor: "_HyARPolicyActionSelector"  # type: ignore

    def __init__(
        self,
        env_or_env_info,
        config: HyARConfig = HyARConfig(),
        critic_builder=DefaultCriticBuilder(),
        critic_solver_builder=DefaultSolverBuilder(),
        actor_builder=DefaultActorBuilder(),
        actor_solver_builder=DefaultActorSolverBuilder(),
        vae_builder=DefaultVAEBuilder(),
        vae_solver_buidler=DefaultVAESolverBuilder(),
        replay_buffer_builder=DefaultReplayBufferBuilder(),
        vae_buffer_builder=DefaultVAEBufferBuilder(),
        explorer_builder=DefaultExplorerBuilder(),
        pretrain_explorer_builder=DefaultPretrainExplorerBuilder(),
    ):
        super().__init__(
            env_or_env_info,
            config,
            critic_builder,
            critic_solver_builder,
            actor_builder,
            actor_solver_builder,
            replay_buffer_builder,
            explorer_builder,
        )

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._vae = vae_builder("vae", self._env_info, self._config)
            self._vae_solver = vae_solver_buidler(self._env_info, self._config)
            # We use different replay buffer for vae
            self._vae_replay_buffer = vae_buffer_builder(env_info=self._env_info, algorithm_config=self._config)
            self._pretrain_explorer_builder = pretrain_explorer_builder

        self._evaluation_actor = _HyARPolicyActionSelector(
            self._env_info,
            self._pi.shallowcopy(),
            self._vae.shallowcopy(),
            embed_dim=self._config.embed_dim,
            latent_dim=self._config.latent_dim,
        )
        self._exploration_actor = _HyARPolicyActionSelector(
            self._env_info,
            self._pi.shallowcopy(),
            self._vae.shallowcopy(),
            embed_dim=self._config.embed_dim,
            latent_dim=self._config.latent_dim,
            append_noise=True,
            sigma=self._config.exploration_noise_sigma,
            action_clip_low=-1.0,
            action_clip_high=1.0,
        )
        self._episode_number = 1
        self._experienced_episodes = 0

    def _before_training_start(self, env_or_buffer):
        super()._before_training_start(env_or_buffer)
        self._vae_trainer = self._setup_vae_training(env_or_buffer)
        self._pretrain_explorer = self._setup_pretrain_explorer(env_or_buffer)
        if isinstance(env_or_buffer, gym.Env):
            self._pretrain_vae(env_or_buffer)

    def _setup_q_function_training(self, env_or_buffer):
        # training input/loss variables
        q_function_trainer_config = MT.q_value_trainers.HyARQTrainerConfig(
            reduction_method="mean",
            q_loss_scalar=1.0,
            grad_clip=None,
            train_action_noise_sigma=self._config.train_action_noise_sigma,
            train_action_noise_abs=self._config.train_action_noise_abs,
            noisy_action_max=self._config.noisy_action_max,
            noisy_action_min=self._config.noisy_action_min,
            num_steps=self._config.num_steps,
            unroll_steps=self._config.critic_unroll_steps,
            burn_in_steps=self._config.critic_burn_in_steps,
            reset_on_terminal=self._config.critic_reset_rnn_on_terminal,
            embed_dim=self._config.embed_dim,
            latent_dim=self._config.latent_dim,
        )
        q_function_trainer = MT.q_value_trainers.HyARQTrainer(
            train_functions=self._train_q_functions,
            solvers=self._train_q_solvers,
            target_functions=self._target_q_functions,
            target_policy=self._target_pi,
            vae=self._vae,
            env_info=self._env_info,
            config=q_function_trainer_config,
        )
        for q, target_q in zip(self._train_q_functions, self._target_q_functions):
            sync_model(q, target_q)
        return q_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        # return super()._setup_policy_training(env_or_buffer)
        action_dim = self._config.latent_dim + self._config.embed_dim
        policy_trainer_config = MT.policy_trainers.HyARPolicyTrainerConfig(
            unroll_steps=self._config.actor_unroll_steps,
            burn_in_steps=self._config.actor_burn_in_steps,
            reset_on_terminal=self._config.actor_reset_rnn_on_terminal,
            p_max=np.ones(shape=(1, action_dim)),
            p_min=-np.ones(shape=(1, action_dim)),
        )
        policy_trainer = MT.policy_trainers.HyARPolicyTrainer(
            models=self._pi,
            solvers={self._pi.scope_name: self._pi_solver},
            q_function=self._q1,
            env_info=self._env_info,
            config=policy_trainer_config,
        )
        sync_model(self._pi, self._target_pi, 1.0)

        return policy_trainer

    def _setup_vae_training(self, env_or_buffer):
        vae_trainer_config = MT.encoder_trainers.HyARVAETrainerConfig(
            unroll_steps=self._config.critic_unroll_steps,
            burn_in_steps=self._config.critic_burn_in_steps,
            reset_on_terminal=self._config.critic_reset_rnn_on_terminal,
        )
        return MT.encoder_trainers.HyARVAETrainer(
            self._vae, {self._vae.scope_name: self._vae_solver}, self._env_info, vae_trainer_config
        )

    def _setup_pretrain_explorer(self, env_or_buffer):
        return (
            None
            if self._is_buffer(env_or_buffer)
            else self._pretrain_explorer_builder(self._env_info, self._config, self)
        )

    def _pretrain_vae(self, env: gym.Env):
        for _ in range(self._config.vae_pretrain_episodes):
            experiences = self._pretrain_explorer.rollout(env)
            self._vae_replay_buffer.append_all(experiences)

        for _ in range(self._config.vae_pretrain_times):
            self._vae_training(self._vae_replay_buffer, self._config.vae_pretrain_batch_size)
        c_rate, ds_rate = self._compute_reconstruction_rate(self._vae_replay_buffer)
        self._c_rate = c_rate
        self._ds_rate = ds_rate
        self._exploration_actor.update_c_rate(c_rate)
        self._evaluation_actor.update_c_rate(c_rate)

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        self._vae_replay_buffer.append_all(experiences)

        (_, _, _, non_terminal, *_) = experiences[-1]
        end_of_episode = non_terminal == 0.0
        if end_of_episode:
            self._experienced_episodes += 1
            if self._experienced_episodes < self._config.noise_decay_steps:
                ratio = self._experienced_episodes / self._config.noise_decay_steps
                new_sigma = (
                    self._config.initial_exploration_noise * (1.0 - ratio)
                    + self._config.final_exploration_noise * ratio
                )
                self._exploration_actor.update_sigma(sigma=new_sigma)
            else:
                self._exploration_actor.update_sigma(sigma=self._config.final_exploration_noise)
        if self._config.start_timesteps < self.iteration_num:
            self._hyar_training(self._replay_buffer, self._vae_replay_buffer, end_of_episode)

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _hyar_training(self, replay_buffer, vae_replay_buffer, end_of_episode=False):
        self._rl_training(replay_buffer)
        if (self._experienced_episodes % self._config.T) == 0 and self._iteration_num > 1000 and end_of_episode:
            for _ in range(self._config.vae_training_times):
                self._vae_training(vae_replay_buffer, self._config.vae_training_batch_size)
            c_rate, ds_rate = self._compute_reconstruction_rate(self._vae_replay_buffer)
            self._c_rate = c_rate
            self._ds_rate = ds_rate
            self._exploration_actor.update_c_rate(c_rate)
            self._evaluation_actor.update_c_rate(c_rate)

    def _rl_training(self, replay_buffer):
        actor_steps = self._config.actor_burn_in_steps + self._config.actor_unroll_steps
        critic_steps = self._config.num_steps + self._config.critic_burn_in_steps + self._config.critic_unroll_steps - 1
        num_steps = max(actor_steps, critic_steps)
        experiences_tuple, info = replay_buffer.sample(self._config.batch_size, num_steps=num_steps)
        if num_steps == 1:
            experiences_tuple = (experiences_tuple,)
        assert len(experiences_tuple) == num_steps

        batch = None
        for experiences in reversed(experiences_tuple):
            (s, a, r, non_terminal, s_next, extra, *_) = marshal_experiences(experiences)
            rnn_states = extra["rnn_states"] if "rnn_states" in extra else {}
            extra.update({"c_rate": self._c_rate, "ds_rate": self._ds_rate})
            batch = TrainingBatch(
                batch_size=self._config.batch_size,
                s_current=s,
                a_current=a,
                gamma=self._config.gamma,
                reward=r,
                non_terminal=non_terminal,
                s_next=s_next,
                extra=extra,
                weight=info["weights"],
                next_step_batch=batch,
                rnn_states=rnn_states,
            )

        self._q_function_trainer_state = self._q_function_trainer.train(batch)
        td_errors = self._q_function_trainer_state["td_errors"]
        replay_buffer.update_priorities(td_errors)

        if self.iteration_num % self._config.d == 0:
            # Optimize actor
            self._policy_trainer_state = self._policy_trainer.train(batch)

            # parameter update
            for q, target_q in zip(self._train_q_functions, self._target_q_functions):
                sync_model(q, target_q, tau=self._config.tau)
            sync_model(self._pi, self._target_pi, tau=self._config.tau)

    def _vae_training(self, replay_buffer, batch_size):
        actor_steps = self._config.actor_burn_in_steps + self._config.actor_unroll_steps
        critic_steps = self._config.num_steps + self._config.critic_burn_in_steps + self._config.critic_unroll_steps - 1
        num_steps = max(actor_steps, critic_steps)
        experiences_tuple, info = replay_buffer.sample(batch_size, num_steps=num_steps)
        if num_steps == 1:
            experiences_tuple = (experiences_tuple,)
        assert len(experiences_tuple) == num_steps

        batch = None
        for experiences in reversed(experiences_tuple):
            (s, a, r, non_terminal, s_next, extra, *_) = marshal_experiences(experiences)
            rnn_states = extra["rnn_states"] if "rnn_states" in extra else {}
            batch = TrainingBatch(
                batch_size=batch_size,
                s_current=s,
                a_current=a,
                gamma=self._config.gamma,
                reward=r,
                non_terminal=non_terminal,
                s_next=s_next,
                extra=extra,
                weight=info["weights"],
                next_step_batch=batch,
                rnn_states=rnn_states,
            )

        self._vae_trainer_state = self._vae_trainer.train(batch)

    def _models(self):
        models = super()._models()
        models.update({self._vae.scope_name: self._vae})
        return models

    def _solvers(self):
        solvers = super()._solvers()
        solvers.update({self._vae.scope_name: self._vae_solver})
        return solvers

    def _compute_reconstruction_rate(self, replay_buffer):
        range_rate = 100 - self._config.latent_select_range
        batch_size = self._config.latent_select_batch_size
        border = int(range_rate * (batch_size / 100))
        experiences, _ = replay_buffer.sample(num_samples=batch_size)
        (s, a, _, _, s_next, *_) = marshal_experiences(experiences)

        if not hasattr(self, "_rate_state_var"):
            from nnabla_rl.utils.misc import create_variable

            self._rate_state_var = create_variable(batch_size, self._env_info.state_shape)
            self._rate_action_var = create_variable(batch_size, self._env_info.action_shape)
            self._rate_next_state_var = create_variable(batch_size, self._env_info.state_shape)

            action1, action2 = self._rate_action_var
            x = action1 if isinstance(self._env_info.action_space[0], gym.spaces.Box) else action2
            latent_distribution, (_, predicted_ds) = self._vae.encode_and_decode(
                x=x, state=self._rate_state_var, action=self._rate_action_var
            )
            z = latent_distribution.sample()
            # NOTE: ascending order
            z_sorted = NF.sort(z, axis=0)
            z_up = z_sorted[batch_size - border - 1, :]
            z_down = z_sorted[border, :]
            z_up.persistent = True
            z_down.persistent = True

            ds = self._rate_next_state_var - self._rate_state_var
            ds_rate = RF.mean_squared_error(ds, predicted_ds)
            ds_rate.persistent = True

            self._ds_rate_var = ds_rate
            self._z_up_var = z_up
            self._z_down_var = z_down

        set_data_to_variable(self._rate_state_var, s)
        set_data_to_variable(self._rate_action_var, a)
        set_data_to_variable(self._rate_next_state_var, s_next)

        nn.forward_all((self._z_up_var, self._z_down_var, self._ds_rate_var), clear_no_need_grad=True)
        return (self._z_up_var.d, self._z_down_var.d), self._ds_rate_var.d

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = (
            EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) else env_or_env_info
        )
        return env_info.is_tuple_action_env() and not env_info.is_tuple_state_env()

    @classmethod
    def is_rnn_supported(self):
        return False

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super().latest_iteration_state
        if hasattr(self, "_vae_trainer_state"):
            latest_iteration_state["scalar"].update(
                {
                    "encoder_loss": float(self._vae_trainer_state["encoder_loss"]),
                    "kl_loss": float(self._vae_trainer_state["kl_loss"]),
                    "reconstruction_loss": float(self._vae_trainer_state["reconstruction_loss"]),
                    "dyn_loss": float(self._vae_trainer_state["dyn_loss"]),
                }
            )
        return latest_iteration_state


class _HyARPolicyActionSelector(_ActionSelector[DeterministicPolicy]):
    _vae: HyARVAE

    def __init__(
        self,
        env_info: EnvironmentInfo,
        model: DeterministicPolicy,
        vae: HyARVAE,
        embed_dim: int,
        latent_dim: int,
        append_noise: bool = False,
        action_clip_low: float = np.finfo(np.float32).min,  # type: ignore
        action_clip_high: float = np.finfo(np.float32).max,  # type: ignore
        sigma: float = 1.0,
    ):
        super().__init__(env_info, model)
        self._vae = vae
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim

        self._e: nn.Variable = None
        self._z: nn.Variable = None

        self._append_noise = append_noise
        self._action_clip_low = action_clip_low
        self._action_clip_high = action_clip_high
        self._sigma = nn.Variable.from_numpy_array(sigma * np.ones(shape=(1, 1)))

        # This value is used in the author's code to modify the action
        z_up = nn.Variable.from_numpy_array(np.ones(shape=(1, self._latent_dim)))
        z_down = nn.Variable.from_numpy_array(-np.ones(shape=(1, self._latent_dim)))
        self._c_rate = (z_up, z_down)

    def __call__(self, s, *, begin_of_episode=False, extra_info={}):
        action, info = super().__call__(s, begin_of_episode=begin_of_episode, extra_info=extra_info)
        # Use only the first item in the batch
        # self._e.d[0] and self._z.d[0]
        e = self._e.d[0]
        z = self._z.d[0]
        info.update({"e": e, "z": z})
        (d_action, c_action) = action
        return (d_action, c_action), info

    def update_sigma(self, sigma):
        self._sigma.d = sigma

    def update_c_rate(self, c_rate):
        self._c_rate[0].d = c_rate[0]
        self._c_rate[1].d = c_rate[1]

    def _compute_action(self, state_var: nn.Variable) -> nn.Variable:
        latent_action = self._model.pi(state_var)
        if self._append_noise:
            noise = NF.randn(shape=latent_action.shape)
            latent_action = latent_action + noise * self._sigma
            latent_action = NF.clip_by_value(latent_action, min=self._action_clip_low, max=self._action_clip_high)
        self._e = latent_action[:, : self._embed_dim]
        self._e.persistent = True
        self._z = latent_action[:, self._embed_dim :]
        self._z.persistent = True
        assert latent_action.shape[-1] == self._embed_dim + self._latent_dim

        d_action = self._vae.decode_discrete_action(self._e)
        c_action, _ = self._vae.decode(self._apply_c_rate(self._z), state=state_var, action=(d_action, None))

        return d_action, c_action

    def _apply_c_rate(self, z):
        median = 0.5 * (self._c_rate[0] - self._c_rate[1])
        offset = self._c_rate[0] - median
        median = NF.reshape(median, shape=(1, -1))
        offset = NF.reshape(offset, shape=(1, -1))
        z = z * median + offset
        return z


class HyARPolicyExplorerConfig(RawPolicyExplorerConfig):
    pass


class HyARPolicyExplorer(RawPolicyExplorer):
    def _warmup_action(self, env, *, begin_of_episode=False):
        return self.action(self._steps, self._state, begin_of_episode=begin_of_episode)


class HyARPretrainExplorerConfig(EnvironmentExplorerConfig):
    pass


class HyARPretrainExplorer(EnvironmentExplorer):
    def __init__(self, env_info: EnvironmentInfo, config: HyARPretrainExplorerConfig = HyARPretrainExplorerConfig()):
        super().__init__(env_info, config)

    def action(self, step: int, state, *, begin_of_episode: bool = False):
        (d_action, c_action), action_info = self._sample_action(self._env_info)
        return (d_action, c_action), action_info

    def _warmup_action(self, env, *, begin_of_episode=False):
        (d_action, c_action), action_info = self._sample_action(self._env_info)
        return (d_action, c_action), action_info

    def _sample_action(self, env_info):
        action_info = {}
        if env_info.is_tuple_action_env():
            action = []
            for a, action_space in zip(env_info.action_space.sample(), env_info.action_space):
                if isinstance(action_space, gym.spaces.Discrete):
                    a = np.asarray(a).reshape((1,))
                action.append(a)
            action = tuple(action)
        else:
            if env_info.is_discrete_action_env():
                action = env_info.action_space.sample()
                action = np.asarray(action).reshape((1,))
            else:
                action = env_info.action_space.sample()
        return action, action_info
