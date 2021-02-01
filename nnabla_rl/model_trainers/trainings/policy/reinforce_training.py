import nnabla as nn

import numpy as np

from nnabla_rl.model_trainers.model_trainer import TrainingVariables, Training, TrainingBatch


class REINFORCETraining(Training):
    _target_return: nn.Variable

    def __init__(self):
        super(REINFORCETraining, self).__init__()

    def setup_batch(self, batch: TrainingBatch):
        target_return = batch.extra['target_return']
        prev_batch_size = self._target_return.shape[0]
        new_batch_size = target_return.shape[0]
        if prev_batch_size != new_batch_size:
            self._target_return = nn.Variable((new_batch_size, 1))
        target_return = np.reshape(target_return, self._target_return.shape)
        self._target_return.d = target_return
        return batch

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        batch_size = training_variables.batch_size
        if not hasattr(self, '_target_return') or self._target_return.shape[0] != batch_size:
            self._target_return = nn.Variable((batch_size, 1))
        return self._target_return
