from typing import cast, Iterable, Dict, Union, Sequence

import nnabla as nn
import nnabla.functions as NF

import numpy as np

from dataclasses import dataclass

from nnabla_rl.model_trainers.model_trainer import \
    TrainerParam, Training, TrainingVariables, ModelTrainer, TrainingBatch
from nnabla_rl.models import Model, RewardFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list


@dataclass
class GAILRewardFunctionTrainerParam(TrainerParam):
    batch_size: int = 1024
    learning_rate: float = 3e-4
    entropy_coef: float = 0.001

    def __post_init__(self):
        self._assert_positive(self.entropy_coef, "entropy_coef")


class GAILRewardFunctionTrainer(ModelTrainer):
    _params: GAILRewardFunctionTrainerParam

    def __init__(self, env_info, params=GAILRewardFunctionTrainerParam()):
        super(GAILRewardFunctionTrainer, self).__init__(env_info, params)
        self._binary_classification_loss = None

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        s_curr_agent = batch.extra['s_current_agent']
        a_curr_agent = batch.extra['a_current_agent']
        s_next_agent = batch.extra['s_next_agent']
        s_curr_expert = batch.extra['s_current_expert']
        a_curr_expert = batch.extra['a_current_expert']
        s_next_expert = batch.extra['s_next_expert']

        training_variables.extra['s_current_expert'].d = s_curr_expert
        training_variables.extra['a_current_expert'].d = a_curr_expert
        training_variables.extra['s_next_expert'].d = s_next_expert
        training_variables.extra['s_current_agent'].d = s_curr_agent
        training_variables.extra['a_current_agent'].d = a_curr_agent
        training_variables.extra['s_next_agent'].d = s_next_agent

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._binary_classification_loss.forward()
        self._binary_classification_loss.backward()
        for solver in solvers.values():
            solver.update()

        return {}

    def _build_training_graph(self, models: Union[Model, Sequence[Model]],
                              training: Training,
                              training_variables: TrainingVariables):
        models = convert_to_list_if_not_list(models)
        models = cast(Sequence[RewardFunction], models)

        self._binary_classification_loss = 0
        for model in models:
            # fake path
            logits_fake = model.r(training_variables.extra['s_current_agent'],
                                  training_variables.extra['a_current_agent'],
                                  training_variables.extra['s_next_agent'])
            fake_loss = NF.mean(NF.sigmoid_cross_entropy(logits_fake, NF.constant(0, logits_fake.shape)))
            # real path
            logits_real = model.r(training_variables.extra['s_current_expert'],
                                  training_variables.extra['a_current_expert'],
                                  training_variables.extra['s_next_expert'])
            real_loss = NF.mean(NF.sigmoid_cross_entropy(logits_real, NF.constant(1, logits_real.shape)))
            # entropy loss
            logits = NF.concatenate(logits_fake, logits_real, axis=0)
            entropy = NF.mean((1. - NF.sigmoid(logits)) * logits - NF.log_sigmoid(logits))
            entropy_loss = - self._params.entropy_coef * entropy  # maximize
            self._binary_classification_loss += fake_loss + real_loss + entropy_loss

    def _setup_training_variables(self, batch_size):
        s_current_agent_var = nn.Variable((batch_size, *self._env_info.state_shape))
        s_next_agent_var = nn.Variable((batch_size, *self._env_info.state_shape))
        s_current_expert_var = nn.Variable((batch_size, *self._env_info.state_shape))
        s_next_expert_var = nn.Variable((batch_size, *self._env_info.state_shape))

        if self._env_info.is_discrete_action_env():
            a_current_agent_var = nn.Variable((batch_size, 1))
            a_current_expert_var = nn.Variable((batch_size, 1))
        else:
            a_current_agent_var = nn.Variable((batch_size, self._env_info.action_dim))
            a_current_expert_var = nn.Variable((batch_size, self._env_info.action_dim))

        variables = {'s_current_expert': s_current_expert_var,
                     'a_current_expert': a_current_expert_var,
                     's_next_expert': s_next_expert_var,
                     's_current_agent': s_current_agent_var,
                     'a_current_agent': a_current_agent_var,
                     's_next_agent': s_next_agent_var}
        training_variables = TrainingVariables(batch_size, extra=variables)

        return training_variables
