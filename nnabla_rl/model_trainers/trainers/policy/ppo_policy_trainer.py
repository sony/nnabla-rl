from typing import Optional, Iterable, Dict

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

from nnabla_rl.model_trainers.model_trainer import TrainerParam, Training, TrainingVariables, ModelTrainer
from nnabla_rl.models import Model, StochasticPolicy


@dataclass
class PPOTrainingVariables(TrainingVariables):
    log_prob: Optional[nn.Variable] = None
    advantage: Optional[nn.Variable] = None
    alpha: Optional[nn.Variable] = None


@dataclass
class PPOPolicyTrainerParam(TrainerParam):
    entropy_coefficient: float = 0.01
    epsilon: float = 0.1


class PPOPolicyTrainer(ModelTrainer):
    '''Proximal Policy Optimization (PPO) style Policy Trainer
    '''

    def __init__(self, env_info, params=PPOPolicyTrainerParam()):
        super(PPOPolicyTrainer, self).__init__(env_info, params)
        self._pi_loss = None

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      experience,
                      training_variables: TrainingVariables,
                      **kwargs):
        (s, a, log_prob, advantage, alpha) = experience

        training_variables.s_current.d = s
        training_variables.a_current.d = a
        training_variables.log_prob.d = log_prob
        training_variables.advantage.d = advantage
        training_variables.alpha.d = alpha

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward(clear_no_need_grad=True)
        self._pi_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        errors = {}
        return errors

    def _build_training_graph(self, models: Iterable[Model],
                              training: Training,
                              training_variables: TrainingVariables):
        if not isinstance(models[0], StochasticPolicy):
            raise ValueError
        self._pi_loss = 0
        for policy in models:
            distribution = policy.pi(training_variables.s_current)
            log_prob_new = distribution.log_prob(training_variables.a_current)
            log_prob_old = training_variables.log_prob
            probability_ratio = NF.exp(log_prob_new - log_prob_old)
            clipped_ratio = NF.clip_by_value(probability_ratio,
                                             1 - self._params.epsilon * training_variables.alpha,
                                             1 + self._params.epsilon * training_variables.alpha)
            lower_bounds = NF.minimum2(probability_ratio * training_variables.advantage,
                                       clipped_ratio * training_variables.advantage)
            clip_loss = NF.mean(lower_bounds)

            entropy = distribution.entropy()
            entropy_loss = NF.mean(entropy)

            self._pi_loss += -clip_loss - self._params.entropy_coefficient * entropy_loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        if self._env_info.is_discrete_action_env():
            action_shape = (batch_size, 1)
        else:
            action_shape = (batch_size, self._env_info.action_dim)
        a_current_var = nn.Variable(action_shape)
        log_prob_var = nn.Variable((batch_size, 1))
        advantage_var = nn.Variable((batch_size, 1))
        alpha_var = nn.Variable((batch_size, 1))

        return PPOTrainingVariables(s_current_var,
                                    a_current_var,
                                    log_prob=log_prob_var,
                                    advantage=advantage_var,
                                    alpha=alpha_var)
