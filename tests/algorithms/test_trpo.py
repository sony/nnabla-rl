import pytest

import nnabla as nn
import nnabla.parametric_functions as NPF
import nnabla.functions as NF
import nnabla.initializer as NI

import numpy as np

import nnabla_rl.environments as E
import nnabla_rl.algorithms as A
from nnabla_rl.algorithms.trpo import _hessian_vector_product, \
    _concat_network_params_in_ndarray, _update_network_params_by_flat_params
from nnabla_rl.utils.optimization import conjugate_gradient
from nnabla_rl.utils.matrices import compute_hessian


class TestComputeHessianVectorProduct():
    def setup_method(self, method):
        nn.clear_parameters()

    def test_compute_hessian_vector_product_by_hand(self):
        state = nn.Variable((1, 2))
        output = NPF.affine(state, 1, w_init=NI.ConstantInitializer(
            value=1.), with_bias=False)

        loss = NF.sum(output**2)
        grads = nn.grad([loss], nn.get_parameters().values())
        flat_grads = grads[0].reshape((-1, ))
        flat_grads.need_grad = True

        def compute_Ax(vec):
            return _hessian_vector_product(flat_grads, nn.get_parameters().values(), vec)

        state_array = np.array([[1.0, 0.25]])
        state.d = state_array
        flat_grads.forward()

        actual = conjugate_gradient(
            compute_Ax, flat_grads.d, max_iterations=None)

        H = np.array(
            [[2*state_array[0, 0]**2,
              2*state_array[0, 0]*state_array[0, 1]],
             [2*state_array[0, 0]*state_array[0, 1],
              2*state_array[0, 1]**2]]
        )
        expected = np.matmul(np.linalg.pinv(H), flat_grads.d.reshape(-1, 1))

        assert expected == pytest.approx(actual.reshape(-1, 1), abs=1e-5)

    def test_compute_hessian_vector_product_by_hessian(self):
        state = nn.Variable((1, 2))
        output = NPF.affine(state, 1, w_init=NI.ConstantInitializer(
            value=1.), b_init=NI.ConstantInitializer(value=1.))

        loss = NF.sum(output**2)
        grads = nn.grad([loss], nn.get_parameters().values())
        flat_grads = NF.concatenate(*[grad.reshape((-1,)) for grad in grads])
        flat_grads.need_grad = True

        def compute_Ax(vec):
            return _hessian_vector_product(flat_grads, nn.get_parameters().values(), vec)

        state_array = np.array([[1.0, 0.5]])
        state.d = state_array
        flat_grads.forward()

        actual = conjugate_gradient(
            compute_Ax, flat_grads.d, max_iterations=None)

        hessian = compute_hessian(loss, nn.get_parameters().values())

        expected = np.matmul(np.linalg.pinv(hessian),
                             flat_grads.d.reshape(-1, 1))

        assert expected == pytest.approx(actual.reshape(-1, 1), abs=1e-5)


class TestConcatNetworkParamsInNdarray():
    def test_concat_network_params_in_ndarray(self):
        nn.clear_parameters()
        np.random.seed(0)

        state = nn.Variable((1, 2))
        output = NPF.affine(state, 1)
        params = nn.get_parameters()

        actual = _concat_network_params_in_ndarray(params)
        state.d = np.random.randn(1, 2)
        output.forward()

        assert len(actual) == 3
        assert np.allclose(params["affine/W"].d.flatten(), actual[:2])
        assert np.allclose(params["affine/b"].d.flatten(), actual[-1])


class TestUpdateNetworkParametersByFlatParams():
    def test_update_network_params_by_flat_params(self):
        nn.clear_parameters()
        np.random.seed(0)

        state = nn.Variable((1, 2))
        output = NPF.affine(state, 1)
        params = nn.get_parameters()
        new_flat_params = np.random.randn(3)

        _update_network_params_by_flat_params(params, new_flat_params)
        state.d = np.random.randn(1, 2)
        output.forward()  # dummy forward

        assert np.allclose(new_flat_params[:2], params["affine/W"].d.flatten())
        assert np.allclose(new_flat_params[-1], params["affine/b"].d.flatten())


class TestTRPO():
    def setup_method(self):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        trpo = A.TRPO(dummy_env)

        assert trpo.__name__ == 'TRPO'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """
        dummy_env = E.DummyContinuous()
        dummy_env = EpisodicEnv(dummy_env, min_episode_length=3)

        params = A.TRPOParam(num_steps_per_iteration=5,
                             vf_batch_size=2,
                             sigma_kl_divergence_constraint=10.0,
                             maximum_backtrack_numbers=50)
        trpo = A.TRPO(dummy_env, params=params)

        trpo.train_online(dummy_env, total_iterations=1)

    def test_old_network_initialization(self):
        dummy_env = E.DummyContinuous()
        trpo = A.TRPO(dummy_env)

        # Should be initialized to same parameters
        assert self._has_same_parameters(
            trpo._policy .get_parameters(), trpo._old_policy.get_parameters())

    def test_run_offline_training(self):
        """
        Check that raising error when calling offline training
        """
        dummy_env = E.DummyContinuous()
        trpo = A.TRPO(dummy_env)

        with pytest.raises(NotImplementedError):
            trpo.train_offline([], total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        trpo = A.TRPO(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = trpo.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_solver_has_correct_parameters(self):
        dummy_env = E.DummyContinuous()
        trpo = A.TRPO(dummy_env)

        v_solver_parms = trpo._solvers()["v_solver"].get_parameters()
        v_function_params = trpo._v_function.get_parameters()

        assert is_same_parameter_id_and_key(v_solver_parms, v_function_params)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.TRPOParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.TRPOParam(num_steps_per_iteration=-1)
        with pytest.raises(ValueError):
            A.TRPOParam(sigma_kl_divergence_constraint=-0.1)
        with pytest.raises(ValueError):
            A.TRPOParam(maximum_backtrack_numbers=-0.1)
        with pytest.raises(ValueError):
            A.TRPOParam(conjugate_gradient_damping=-0.1)
        with pytest.raises(ValueError):
            A.TRPOParam(conjugate_gradient_iterations=-5)
        with pytest.raises(ValueError):
            A.TRPOParam(vf_epochs=-5)
        with pytest.raises(ValueError):
            A.TRPOParam(vf_batch_size=-5)
        with pytest.raises(ValueError):
            A.TRPOParam(vf_learning_rate=-0.5)

    def _has_same_parameters(self, params1, params2):
        for key in params1.keys():
            if not np.allclose(params1[key].data.data, params2[key].data.data):
                return False
        return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import EpisodicEnv, is_same_parameter_id_and_key
    pytest.main()
else:
    from .testing_utils import EpisodicEnv, is_same_parameter_id_and_key
