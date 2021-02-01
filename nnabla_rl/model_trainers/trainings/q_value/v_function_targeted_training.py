from typing import cast, Sequence, Union

import nnabla as nn

import nnabla_rl.functions as RNF
from nnabla_rl.model_trainers.model_trainer import Training, TrainingVariables
from nnabla_rl.models import QFunction, VFunction, Model
from nnabla_rl.utils.data import convert_to_list_if_not_list


class _QFunctionVFunctionTargetedTraining(Training):
    _target_functions: Sequence[VFunction]

    def __init__(self, target_functions: Sequence[VFunction]):
        self._target_functions = target_functions

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        target_vs = []
        for v_function in self._target_functions:
            target_vs.append(v_function.v(s_next))
        target_v = RNF.minimum_n(target_vs)
        return reward + gamma * non_terminal * target_v


class VFunctionTargetedTraining(Training):
    _delegate: Training

    def __init__(self,
                 train_functions: Union[Model, Sequence[Model]],
                 target_functions: Union[Model, Sequence[Model]]):
        train_functions = convert_to_list_if_not_list(train_functions)
        target_functions = convert_to_list_if_not_list(target_functions)

        train_function = train_functions[0]
        target_function = target_functions[0]
        if isinstance(train_function, QFunction) and isinstance(target_function, VFunction):
            target_functions = cast(Sequence[VFunction], target_functions)
            self._delegate = _QFunctionVFunctionTargetedTraining(target_functions)
        else:
            raise NotImplementedError(f'No training implementation for class: {train_function.__class__}')

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        return self._delegate.compute_target(training_variables, **kwargs)
