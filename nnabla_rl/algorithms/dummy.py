from nnabla_rl.algorithm import Algorithm, AlgorithmParam

from nnabla_rl.logger import logger
import nnabla_rl as rl


class DummyParam(AlgorithmParam):
    pass


class Dummy(Algorithm):
    """
    This algorithm does nothing. Just used for understanding the concept of the framework.
    """

    def __init__(self, env_or_env_info, params=DummyParam()):
        super(Dummy, self).__init__(env_or_env_info, params=params)
        self._action_space = self._env_info.action_space

    def compute_eval_action(self, state):
        return self._action_space.sample()

    def _build_evaluation_graph(self):
        assert rl.is_eval_scope()

    def _before_training_start(self, env_or_buffer):
        logger.debug("Before training start!! Write your algorithm's initializations here.")

    def _run_online_training_iteration(self, env):
        logger.debug("Running online training loop. Iteartion: {}".format(
            self.iteration_num))

    def _run_offline_training_iteration(self, buffer):
        logger.debug("Running offline training loop. Iteartion: {}".format(
            self.iteration_num))

    def _after_training_finish(self, env_or_buffer):
        logger.debug("Training finished. Do your algorithm's finalizations here.")

    def _models(self):
        return {}

    def _solvers(self):
        return {}
