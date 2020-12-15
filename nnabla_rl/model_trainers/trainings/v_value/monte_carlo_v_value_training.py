import nnabla as nn

from nnabla_rl.model_trainers.model_trainer import TrainingVariables, Training, TrainingBatch


class MonteCarloVValueTraining(Training):
    def __init__(self):
        super(MonteCarloVValueTraining, self).__init__()
        self._v_target = None

    def setup_batch(self, batch: TrainingBatch):
        batch_size = batch.batch_size
        v_target = batch.extra['v_target']
        if self._v_target is None or self._v_target.shape[0] != batch_size:
            self._v_target = nn.Variable((batch_size, 1))
        self._v_target.d = v_target
        return batch

    def compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        batch_size = training_variables.batch_size
        if self._v_target is None or self._v_target.shape[0] != batch_size:
            self._v_target = nn.Variable((batch_size, 1))
        return self._v_target
