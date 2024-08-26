# Copyright 2024 Sony Group Corporation.
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

import pytest

import nnabla as nn
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.models import VFunction


class DummyVFunction(VFunction):
    def __init__(self):
        super(DummyVFunction, self).__init__("test_v_function")

    def v(self, s):
        with nn.parameter_scope(self.scope_name):
            if isinstance(s, tuple):
                h = s[0] * 2.0 + s[1] * 2.0
            else:
                h = s * 2.0
        return h


class TestOptionCritic(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        option_critic = A.OptionCritic(dummy_env)

        assert option_critic.__name__ == "OptionCritic"

    def test_run_online_training(self):
        """Check that no error occurs when calling online training (amp env)"""

        dummy_env = E.DummyDiscreteImg()
        config = A.OptionCriticConfig(
            option_v_batch_size=3, start_timesteps=5, target_update_frequency=1, learner_update_frequency=1
        )
        option_critic = A.OptionCritic(dummy_env, config=config)
        option_critic.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """Check that no error occurs when calling offline training."""

        dummy_env = E.DummyDiscreteImg()
        option_critic = A.OptionCritic(dummy_env)

        with pytest.raises(NotImplementedError):
            option_critic.train_offline([], total_iterations=10)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.OptionCriticConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(gamma=-0.1)

        with pytest.raises(ValueError):
            A.OptionCriticConfig(intra_policy_learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(termination_function_learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(option_v_function_learning_rate=-0.1)

        with pytest.raises(ValueError):
            A.OptionCriticConfig(option_v_batch_size=-1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(termination_function_batch_size=-1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(intra_policy_batch_size=-1)

        with pytest.raises(ValueError):
            A.OptionCriticConfig(learner_update_frequency=-1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(target_update_frequency=-1)

        with pytest.raises(ValueError):
            A.OptionCriticConfig(start_timesteps=-1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(replay_buffer_size=-1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(start_timesteps=100, replay_buffer_size=10)

        with pytest.raises(ValueError):
            A.OptionCriticConfig(max_option_explore_steps=-1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(initial_option_epsilon=-0.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(initial_option_epsilon=1.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(final_option_epsilon=-0.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(final_option_epsilon=1.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(test_option_epsilon=-0.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(test_option_epsilon=1.1)

        with pytest.raises(ValueError):
            A.OptionCriticConfig(advantage_offset=-0.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(entropy_regularizer_coefficient=-0.1)
        with pytest.raises(ValueError):
            A.OptionCriticConfig(num_options=-1.0)

    def test_latest_iteration_state(self):
        """Check that latest iteration state has the keys and values we
        expected."""

        dummy_env = E.DummyDiscreteImg()
        option_critic = A.OptionCritic(dummy_env)

        option_critic._option_v_function_trainer_state = {"option_v_loss": 0.0}
        option_critic._termination_function_trainer_state = {"termination_loss": 1.0}
        option_critic._intra_policy_trainer_state = {"intra_pi_loss": 2.0}

        latest_iteration_state = option_critic.latest_iteration_state
        assert "option_v_loss" in latest_iteration_state["scalar"]
        assert "termination_loss" in latest_iteration_state["scalar"]
        assert "intra_pi_loss" in latest_iteration_state["scalar"]
        assert latest_iteration_state["scalar"]["option_v_loss"] == 0.0
        assert latest_iteration_state["scalar"]["termination_loss"] == 1.0
        assert latest_iteration_state["scalar"]["intra_pi_loss"] == 2.0


if __name__ == "__main__":
    pytest.main()
