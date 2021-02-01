from typing import cast, Sequence, Union

import nnabla as nn

import nnabla_rl.functions as RNF
from nnabla_rl.model_trainers.model_trainer import Training, TrainingVariables
from nnabla_rl.models import QFunction, StochasticPolicy, Model
from nnabla_rl.utils.data import convert_to_list_if_not_list


class _QFunctionSoftQTraining(Training):
    _target_functions: Sequence[QFunction]
    _target_policy: StochasticPolicy
    _temperature: nn.Variable

    def __init__(self,
                 target_functions: Sequence[QFunction],
                 target_policy: StochasticPolicy,
                 temperature: nn.Variable):
        self._target_functions = target_functions
        self._target_policy = target_policy
        self._temperature = temperature

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        policy_distribution = self._target_policy.pi(s_next)
        a_next, log_pi = policy_distribution.sample_and_compute_log_prob()

        q_values = []
        for target_q_function in self._target_functions:
            q_value = target_q_function.q(s_next, a_next)
            q_values.append(q_value)
        target_q = RNF.minimum_n(q_values)
        return reward + gamma * non_terminal * (target_q - self._temperature * log_pi)


class SoftQTraining(Training):
    def __init__(self,
                 train_functions: Union[Model, Sequence[Model]],
                 target_functions: Union[Model, Sequence[Model]],
                 target_policy: StochasticPolicy,
                 temperature: nn.Variable):
        train_functions = convert_to_list_if_not_list(train_functions)
        target_functions = convert_to_list_if_not_list(target_functions)
        train_function = train_functions[0]
        target_function = target_functions[0]
        if isinstance(train_function, QFunction) and isinstance(target_function, QFunction):
            target_functions = cast(Sequence[QFunction], target_functions)
            self._delegate = _QFunctionSoftQTraining(target_functions, target_policy, temperature)
        else:
            raise NotImplementedError(f'No training implementation for class: {target_function.__class__}')

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        return self._delegate.compute_target(training_variables, **kwargs)
