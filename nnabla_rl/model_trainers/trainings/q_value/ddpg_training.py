from typing import Iterable, Union

import nnabla as nn

import nnabla_rl.functions as RNF
from nnabla_rl.model_trainers.model_trainer import Training, TrainingVariables
from nnabla_rl.models import QFunction, DeterministicPolicy, Model
from nnabla_rl.utils.data import convert_to_list_if_not_iterable


class _QFunctionDDPGTraining(Training):
    def __init__(self,
                 target_functions: Iterable[QFunction],
                 target_policy: DeterministicPolicy):
        self._target_functions = target_functions
        self._target_policy = target_policy

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        q_values = []
        a_next = self._target_policy.pi(s_next)
        a_next.need_grad = False
        for target_q_function in self._target_functions:
            q_value = target_q_function.q(s_next, a_next)
            q_values.append(q_value)
        # Use the minimum among computed q_values by default
        target_q = RNF.minimum_n(q_values)
        return reward + gamma * non_terminal * target_q


class DDPGTraining(Training):
    def __init__(self,
                 train_functions: Union[Model, Iterable[Model]],
                 target_functions: Union[Model, Iterable[Model]],
                 target_policy: DeterministicPolicy):
        train_functions = convert_to_list_if_not_iterable(train_functions)
        target_functions = convert_to_list_if_not_iterable(target_functions)

        train_function = train_functions[0]
        target_function = target_functions[0]
        if isinstance(train_function, QFunction) and isinstance(target_function, QFunction):
            self._delegate = _QFunctionDDPGTraining(target_functions, target_policy)
        else:
            raise NotImplementedError(f'No training implementation for class: {train_function.__class__}')

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        return self._delegate.compute_target(training_variables, **kwargs)
