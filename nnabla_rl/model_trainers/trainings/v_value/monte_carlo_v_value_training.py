import nnabla as nn

from nnabla_rl.model_trainers.model_trainer import TrainingVariables, Training


class MonteCarloVValueTraining(Training):
    def __init__(self):
        super(MonteCarloVValueTraining, self).__init__()
        self._target_value = None

    def setup_experience(self, experience):
        # FIXME: The order of values provided in the experience should be defined properly
        (_, target_value, *_) = experience
        batch_size = target_value.shape[0]
        if self._target_value is None or self._target_value.shape[0] != batch_size:
            self._target_value = nn.Variable((batch_size, 1))
        self._target_value.d = target_value
        return experience

    def compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        batch_size = training_variables.batch_size
        if self._target_value is None or self._target_value.shape[0] != batch_size:
            self._target_value = nn.Variable((batch_size, 1))
        return self._target_value
