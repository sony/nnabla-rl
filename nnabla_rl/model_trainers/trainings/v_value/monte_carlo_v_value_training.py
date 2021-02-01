import nnabla as nn

from nnabla_rl.model_trainers.model_trainer import TrainingVariables, Training, TrainingBatch


class MonteCarloVValueTraining(Training):
    _v_target: nn.Variable

    def __init__(self):
        super(MonteCarloVValueTraining, self).__init__()

    def setup_batch(self, batch: TrainingBatch):
        batch_size = batch.batch_size
        v_target = batch.extra['v_target']
        if self._v_target is None or self._v_target.shape[0] != batch_size:
            self._v_target = nn.Variable((batch_size, 1))
        self._v_target.d = v_target
        return batch

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        batch_size = training_variables.batch_size
        if not hasattr(self, '_v_target') or self._v_target.shape[0] != batch_size:
            self._v_target = nn.Variable((batch_size, 1))
        return self._v_target
