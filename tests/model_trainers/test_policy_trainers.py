# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

from abc import ABCMeta

import numpy as np
import pytest

import nnabla as nn
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
import nnabla_rl.model_trainers as MT
from nnabla_rl.distributions.gaussian import Gaussian
from nnabla_rl.environments.dummy import DummyContinuous
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import LossIntegration
from nnabla_rl.model_trainers.policy.dpg_policy_trainer import DPGPolicyTrainer
from nnabla_rl.model_trainers.policy.soft_policy_trainer import AdjustableTemperature, SoftPolicyTrainer
from nnabla_rl.model_trainers.policy.trpo_policy_trainer import (
    _concat_network_params_in_ndarray,
    _hessian_vector_product,
    _update_network_params_by_flat_params,
)
from nnabla_rl.models import TD3Policy, TD3QFunction
from nnabla_rl.models.mujoco.policies import SACPolicy
from nnabla_rl.models.mujoco.q_functions import SACQFunction
from nnabla_rl.models.policy import DeterministicPolicy, StochasticPolicy
from nnabla_rl.utils.matrices import compute_hessian
from nnabla_rl.utils.optimization import conjugate_gradient


class DeterministicRnnPolicy(DeterministicPolicy):
    def __init__(self, scope_name: str):
        super().__init__(scope_name)
        self._internal_state_shape = (10,)
        self._fake_internal_state = None

    def is_recurrent(self) -> bool:
        return True

    def internal_state_shapes(self):
        return {"fake": self._internal_state_shape}

    def set_internal_states(self, states):
        self._fake_internal_state = states["fake"]

    def get_internal_states(self):
        return {"fake": self._fake_internal_state}

    def pi(self, s):
        self._fake_internal_state = self._fake_internal_state * 2
        return s


class StochasticRnnPolicy(StochasticPolicy):
    def __init__(self, scope_name: str):
        super().__init__(scope_name)
        self._internal_state_shape = (10,)
        self._fake_internal_state = None

    def is_recurrent(self) -> bool:
        return True

    def internal_state_shapes(self):
        return {"fake": self._internal_state_shape}

    def set_internal_states(self, states):
        self._fake_internal_state = states["fake"]

    def get_internal_states(self):
        return {"fake": self._fake_internal_state}

    def pi(self, s):
        self._fake_internal_state = self._fake_internal_state * 2
        return Gaussian(
            mean=nn.Variable.from_numpy_array(np.zeros(s.shape)), ln_var=nn.Variable.from_numpy_array(np.ones(s.shape))
        )


class TrainerTest(metaclass=ABCMeta):
    def setup_method(self, method):
        nn.clear_parameters()
        np.random.seed(0)


class TestBEARPolicyTrainer(TrainerTest):
    def test_compute_gaussian_mmd(self):
        def gaussian_kernel(x):
            return x**2

        samples1 = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[2, 2, 2], [2, 2, 2], [3, 3, 3]]], dtype=np.float32)
        samples2 = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 2, 3], [1, 1, 1]]], dtype=np.float32)
        samples1_var = nn.Variable(samples1.shape)
        samples1_var.d = samples1
        samples2_var = nn.Variable(samples2.shape)
        samples2_var.d = samples2

        actual_mmd = MT.policy_trainers.bear_policy_trainer._compute_gaussian_mmd(
            samples1=samples1_var, samples2=samples2_var, sigma=20.0
        )
        actual_mmd.forward()
        expected_mmd = self._compute_mmd(samples1, samples2, sigma=20.0, kernel=gaussian_kernel)

        assert actual_mmd.shape == (samples1.shape[0], 1)
        assert np.all(np.isclose(actual_mmd.d, expected_mmd))

    def test_compute_laplacian_mmd(self):
        def laplacian_kernel(x):
            return np.abs(x)

        samples1 = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[2, 2, 2], [2, 2, 2], [3, 3, 3]]], dtype=np.float32)
        samples2 = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 2, 3], [1, 1, 1]]], dtype=np.float32)
        samples1_var = nn.Variable(samples1.shape)
        samples1_var.d = samples1
        samples2_var = nn.Variable(samples2.shape)
        samples2_var.d = samples2

        actual_mmd = MT.policy_trainers.bear_policy_trainer._compute_laplacian_mmd(
            samples1=samples1_var, samples2=samples2_var, sigma=20.0
        )
        actual_mmd.forward()
        expected_mmd = self._compute_mmd(samples1, samples2, sigma=20.0, kernel=laplacian_kernel)

        assert actual_mmd.shape == (samples1.shape[0], 1)
        assert np.all(np.isclose(actual_mmd.d, expected_mmd))

    def _compute_mmd(self, samples1, samples2, sigma, kernel):
        diff_xx = self._compute_kernel_sum(samples1, samples1, sigma, kernel)
        diff_xy = self._compute_kernel_sum(samples1, samples2, sigma, kernel)
        diff_yy = self._compute_kernel_sum(samples2, samples2, sigma, kernel)
        n = samples1.shape[1]
        m = samples2.shape[1]
        mmd = diff_xx / (n * n) - 2.0 * diff_xy / (n * m) + diff_yy / (m * m)
        mmd = np.sqrt(mmd + 1e-6)
        return mmd

    def _compute_kernel_sum(self, a, b, sigma, kernel):
        sums = []
        for index in range(a.shape[0]):
            kernel_sum = 0.0
            for i in range(a.shape[1]):
                for j in range(b.shape[1]):
                    diff = 0.0
                    for k in range(a.shape[2]):
                        # print(f'samples[{i}] - samples[{j}]={samples1[i]-samples1[j]}')
                        diff += kernel(a[index][i][k] - b[index][j][k])
                    kernel_sum += np.exp(-diff / (2.0 * sigma))
            sums.append(kernel_sum)
        return np.reshape(np.array(sums), newshape=(len(sums), 1))


class TestComputeHessianVectorProduct(TrainerTest):
    def test_compute_hessian_vector_product_by_hand(self):
        state = nn.Variable((1, 2))
        output = NPF.affine(state, 1, w_init=NI.ConstantInitializer(value=1.0), with_bias=False)

        loss = NF.sum(output**2)
        grads = nn.grad([loss], nn.get_parameters().values())
        flat_grads = grads[0].reshape((-1,))
        flat_grads.need_grad = True

        def compute_Ax(vec):
            return _hessian_vector_product(flat_grads, nn.get_parameters().values(), vec)

        state_array = np.array([[1.0, 0.25]])
        state.d = state_array
        flat_grads.forward()

        actual = conjugate_gradient(compute_Ax, flat_grads.d, max_iterations=1000)

        H = np.array(
            [
                [2 * state_array[0, 0] ** 2, 2 * state_array[0, 0] * state_array[0, 1]],
                [2 * state_array[0, 0] * state_array[0, 1], 2 * state_array[0, 1] ** 2],
            ]
        )
        expected = np.matmul(np.linalg.pinv(H), flat_grads.d.reshape(-1, 1))

        assert expected == pytest.approx(actual.reshape(-1, 1), abs=1e-5)

    def test_compute_hessian_vector_product_by_hessian(self):
        state = nn.Variable((1, 2))
        output = NPF.affine(
            state, 1, w_init=NI.ConstantInitializer(value=1.0), b_init=NI.ConstantInitializer(value=1.0)
        )

        loss = NF.sum(output**2)
        grads = nn.grad([loss], nn.get_parameters().values())
        flat_grads = NF.concatenate(*[grad.reshape((-1,)) for grad in grads])
        flat_grads.need_grad = True

        def compute_Ax(vec):
            return _hessian_vector_product(flat_grads, nn.get_parameters().values(), vec)

        state_array = np.array([[1.0, 0.5]])
        state.d = state_array
        flat_grads.forward()

        actual = conjugate_gradient(compute_Ax, flat_grads.d, max_iterations=1000)

        hessian = compute_hessian(loss, nn.get_parameters().values())

        expected = np.matmul(np.linalg.pinv(hessian), flat_grads.d.reshape(-1, 1))

        assert expected == pytest.approx(actual.reshape(-1, 1), abs=1e-5)


class TestConcatNetworkParamsInNdarray(TrainerTest):
    def test_concat_network_params_in_ndarray(self):
        state = nn.Variable((1, 2))
        output = NPF.affine(state, 1)
        params = nn.get_parameters()

        actual = _concat_network_params_in_ndarray(params)
        state.d = np.random.randn(1, 2)
        output.forward()

        assert len(actual) == 3
        assert np.allclose(params["affine/W"].d.flatten(), actual[:2])
        assert np.allclose(params["affine/b"].d.flatten(), actual[-1])


class TestUpdateNetworkParametersByFlatParams(TrainerTest):
    def test_update_network_params_by_flat_params(self):
        state = nn.Variable((1, 2))
        output = NPF.affine(state, 1)
        params = nn.get_parameters()
        new_flat_params = np.random.randn(3)

        _update_network_params_by_flat_params(params, new_flat_params)
        state.d = np.random.randn(1, 2)
        output.forward()  # dummy forward

        assert np.allclose(new_flat_params[:2], params["affine/W"].d.flatten())
        assert np.allclose(new_flat_params[-1], params["affine/b"].d.flatten())


class TestTRPOPolicyTrainer(TrainerTest):
    pass


class TestDPGPolicyTrainer(TrainerTest):
    def setup_method(self, method):
        nn.clear_parameters()

    @pytest.mark.parametrize("unroll_steps", [1, 2])
    @pytest.mark.parametrize("burn_in_steps", [0, 1, 2])
    @pytest.mark.parametrize("loss_integration", [LossIntegration.LAST_TIMESTEP_ONLY, LossIntegration.ALL_TIMESTEPS])
    def test_with_non_rnn_model(self, unroll_steps, burn_in_steps, loss_integration):
        env_info = EnvironmentInfo.from_env(DummyContinuous())

        policy = TD3Policy("stub_pi", action_dim=env_info.action_dim, max_action_value=1)
        train_q = TD3QFunction("stub_q", None)
        # Using DQN Q trainer as representative trainer
        config = MT.policy_trainers.DPGPolicyTrainerConfig(
            unroll_steps=unroll_steps, burn_in_steps=burn_in_steps, loss_integration=loss_integration
        )
        DPGPolicyTrainer(models=policy, q_function=train_q, solvers={}, env_info=env_info, config=config)

        # pass: If no ecror occurs

    @pytest.mark.parametrize("unroll_steps", [1, 2])
    @pytest.mark.parametrize("burn_in_steps", [0, 1, 2])
    @pytest.mark.parametrize("loss_integration", [LossIntegration.LAST_TIMESTEP_ONLY, LossIntegration.ALL_TIMESTEPS])
    def test_with_rnn_model(self, unroll_steps, burn_in_steps, loss_integration):
        env_info = EnvironmentInfo.from_env(DummyContinuous())

        policy = DeterministicRnnPolicy("stub_pi")
        train_q = TD3QFunction("stub_q", None)
        # Using DQN Q trainer as representative trainer
        config = MT.policy_trainers.DPGPolicyTrainerConfig(
            unroll_steps=unroll_steps, burn_in_steps=burn_in_steps, loss_integration=loss_integration
        )
        DPGPolicyTrainer(models=policy, q_function=train_q, solvers={}, env_info=env_info, config=config)

        # pass: If no ecror occurs


class TestSoftPolicyTrainer(TrainerTest):
    def setup_method(self, method):
        nn.clear_parameters()

    @pytest.mark.parametrize("unroll_steps", [1, 2])
    @pytest.mark.parametrize("burn_in_steps", [0, 1, 2])
    @pytest.mark.parametrize("loss_integration", [LossIntegration.LAST_TIMESTEP_ONLY, LossIntegration.ALL_TIMESTEPS])
    def test_with_non_rnn_model(self, unroll_steps, burn_in_steps, loss_integration):
        env_info = EnvironmentInfo.from_env(DummyContinuous())

        policy = SACPolicy("stub_pi", action_dim=env_info.action_dim)
        train_q1 = SACQFunction("stub_q1", None)
        train_q2 = SACQFunction("stub_q2", None)
        # Using DQN Q trainer as representative trainer
        config = MT.policy_trainers.SoftPolicyTrainerConfig(
            unroll_steps=unroll_steps,
            burn_in_steps=burn_in_steps,
            loss_integration=loss_integration,
            fixed_temperature=True,
        )
        SoftPolicyTrainer(
            policy,
            solvers={},
            q_functions=[train_q1, train_q2],
            temperature=AdjustableTemperature("stub_t"),
            temperature_solver=None,
            env_info=env_info,
            config=config,
        )

        # pass: If no ecror occurs

    @pytest.mark.parametrize("unroll_steps", [1, 2])
    @pytest.mark.parametrize("burn_in_steps", [0, 1, 2])
    @pytest.mark.parametrize("loss_integration", [LossIntegration.LAST_TIMESTEP_ONLY, LossIntegration.ALL_TIMESTEPS])
    def test_with_rnn_model(self, unroll_steps, burn_in_steps, loss_integration):
        env_info = EnvironmentInfo.from_env(DummyContinuous())

        policy = StochasticRnnPolicy("stub_pi")
        train_q1 = SACQFunction("stub_q1", None)
        train_q2 = SACQFunction("stub_q2", None)
        config = MT.policy_trainers.SoftPolicyTrainerConfig(
            unroll_steps=unroll_steps,
            burn_in_steps=burn_in_steps,
            loss_integration=loss_integration,
            fixed_temperature=True,
        )
        SoftPolicyTrainer(
            policy,
            solvers={},
            q_functions=[train_q1, train_q2],
            temperature=AdjustableTemperature("stub_t"),
            temperature_solver=None,
            env_info=env_info,
            config=config,
        )

        # pass: If no ecror occurs


class TestAdjustableTemperature(TrainerTest):
    def test_initial_temperature(self):
        initial_value = 5.0
        temperature = AdjustableTemperature(scope_name="test", initial_value=initial_value)
        actual_value = temperature()
        actual_value.forward(clear_no_need_grad=True)

        assert actual_value.data.data == initial_value

        # Create tempearture with random initial value
        nn.clear_parameters()
        temperature = AdjustableTemperature(scope_name="test")
        actual_value = temperature()
        actual_value.forward(clear_no_need_grad=True)

        # No error occurs -> pass

    def test_temperature_is_adjustable(self):
        initial_value = 5.0
        temperature = AdjustableTemperature(scope_name="test", initial_value=initial_value)
        solver = nn.solvers.Adam(alpha=1.0)
        solver.set_parameters(temperature.get_parameters())

        value = temperature()

        loss = 0.5 * NF.mean(value**2)
        loss.forward()

        solver.zero_grad()
        loss.backward()

        solver.update()

        updated_value = temperature()
        updated_value.forward(clear_no_need_grad=True)

        new_value = updated_value.data.data
        assert not np.isclose(new_value, initial_value)
        assert new_value < initial_value


if __name__ == "__main__":
    pytest.main()
