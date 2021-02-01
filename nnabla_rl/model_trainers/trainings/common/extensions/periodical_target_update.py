from typing import Sequence, Union

from nnabla_rl.model_trainers.model_trainer import Training, TrainingExtension
from nnabla_rl.models import Model
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.utils.data import convert_to_list_if_not_list


class PeriodicalTargetUpdate(TrainingExtension):
    _src_models: Sequence[Model]
    _dst_models: Sequence[Model]
    _target_update_frequency: int
    _tau: float

    def __init__(self,
                 training: Training,
                 src_models: Union[Sequence[Model], Model],
                 dst_models: Union[Sequence[Model], Model],
                 target_update_frequency: int = 1,
                 tau: float = 1.0):
        super(PeriodicalTargetUpdate, self).__init__(training)
        self._src_models = convert_to_list_if_not_list(src_models)
        self._dst_models = convert_to_list_if_not_list(dst_models)
        self._target_update_frequency = target_update_frequency
        self._tau = tau

    def after_update(self, train_count: int):
        self._training.after_update(train_count)
        if train_count % self._target_update_frequency == 0:
            for src, dst in zip(self._src_models, self._dst_models):
                copy_network_parameters(src.get_parameters(), dst.get_parameters(), tau=self._tau)
