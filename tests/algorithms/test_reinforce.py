import pytest

import nnabla as nn

import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestREINFORCE(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscrete()
        reinforce = A.REINFORCE(dummy_env)

        assert reinforce.__name__ == 'REINFORCE'

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyDiscrete()
        dummy_env = EpisodicEnv(dummy_env)
        reinforce = A.REINFORCE(dummy_env)
        reinforce.train_online(dummy_env, total_iterations=1)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        dummy_env = E.DummyDiscrete()
        reinforce = A.REINFORCE(dummy_env)

        with pytest.raises(NotImplementedError):
            reinforce.train_offline([], total_iterations=2)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.REINFORCEParam(reward_scale=-0.1)
        with pytest.raises(ValueError):
            A.REINFORCEParam(num_rollouts_per_train_iteration=-1)
        with pytest.raises(ValueError):
            A.REINFORCEParam(learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.REINFORCEParam(clip_grad_norm=-0.1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import EpisodicEnv
    pytest.main()
else:
    from .testing_utils import EpisodicEnv
