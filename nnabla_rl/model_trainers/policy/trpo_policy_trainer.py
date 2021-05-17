# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.logger import logger
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import Model, StochasticPolicy
from nnabla_rl.utils.misc import copy_network_parameters
from nnabla_rl.utils.optimization import conjugate_gradient


def _hessian_vector_product(flat_grads, params, vector):
    ''' Compute multiplied vector hessian of parameters and vector

    Args:
        flat_grads (nn.Variable): gradient of parameters, should be flattened
        params (list[nn.Variable]): list of parameters
        vector (numpy.ndarray): input vector, shape is the same as flat_grads
    Returns:
        hessian_vector (numpy.ndarray): multiplied vector of hessian of parameters and vector
    See:
        https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/
    '''
    assert flat_grads.shape[0] == len(vector)
    if isinstance(vector, np.ndarray):
        vector = nn.Variable.from_numpy_array(vector)
    hessian_multiplied_vector_loss = NF.sum(flat_grads * vector)
    hessian_multiplied_vector_loss.forward()
    for param in params:
        param.grad.zero()
    hessian_multiplied_vector_loss.backward()
    hessian_multiplied_vector = [param.g.copy().flatten() for param in params]
    return np.concatenate(hessian_multiplied_vector)


def _concat_network_params_in_ndarray(params):
    ''' Concatenate network parameters in numpy.ndarray,
        this function returns copied parameters

    Args:
        params (OrderedDict): parameters
    Returns:
        flat_params (numpy.ndarray): flatten parameters in numpy.ndarray type
    '''
    flat_params = []
    for param in params.values():
        flat_param = param.d.copy().flatten()
        flat_params.append(flat_param)
    return np.concatenate(flat_params)


def _update_network_params_by_flat_params(params, new_flat_params):
    ''' Update Network parameters by hand

    Args:
        params (OrderedDict): parameteres
        new_flat_params (numpy.ndarray): flattened new parameters
    '''
    if not isinstance(new_flat_params, np.ndarray):
        raise ValueError("Invalid new_flat_params")
    total_param_numbers = 0
    for param in params.values():
        param_shape = param.shape
        param_numbers = len(param.d.flatten())
        new_param = new_flat_params[total_param_numbers:total_param_numbers +
                                    param_numbers].reshape(param_shape)
        param.d = new_param
        total_param_numbers += param_numbers
    assert total_param_numbers == len(new_flat_params)


@dataclass
class TRPOPolicyTrainerConfig(TrainerConfig):
    gpu_batch_size: Optional[int] = None
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.1
    conjugate_gradient_iterations: int = 20

    def __post_init__(self):
        self._assert_positive(self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(self.conjugate_gradient_damping, 'conjugate_gradient_damping')
        self._assert_positive(self.conjugate_gradient_iterations, 'conjugate_gradient_iterations')


class TRPOPolicyTrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: TRPOPolicyTrainerConfig
    _approximate_return: nn.Variable
    _approximate_return_flat_grads: nn.Variable
    _kl_divergence: nn.Variable
    _kl_divergence_flat_grads: nn.Variable
    _old_policy: StochasticPolicy

    def __init__(self,
                 model: StochasticPolicy,
                 env_info: EnvironmentInfo,
                 config: TRPOPolicyTrainerConfig = TRPOPolicyTrainerConfig()):
        super(TRPOPolicyTrainer, self).__init__(model, {}, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        s = batch.s_current
        a = batch.a_current
        advantage = batch.extra['advantage']

        policy = models[0]

        old_policy = self._old_policy

        full_step_params_update = self._compute_full_step_params_update(
            policy, s, a, advantage, training_variables)

        self._linesearch_and_update_params(policy, s, a, advantage,
                                           full_step_params_update, training_variables)

        copy_network_parameters(policy.get_parameters(), old_policy.get_parameters(), tau=1.0)

        trainer_state: Dict[str, Any] = {}
        return trainer_state

    def _gpu_batch_size(self, batch_size):
        # We use gpu_batch_size to reduce one forward gpu calculation memory
        # Usually, gpu_batch_size is the same as batch_size
        if self._config.gpu_batch_size is None:
            return batch_size
        if self._config.gpu_batch_size < batch_size:
            return self._config.gpu_batch_size
        else:
            return batch_size

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        if len(models) != 1:
            raise RuntimeError('TRPO training only support 1 model for training')
        models = cast(Sequence[StochasticPolicy], models)
        policy = models[0]
        if not hasattr(self, '_old_policy'):
            self._old_policy = cast(StochasticPolicy, policy.deepcopy('old_policy'))
        old_policy = self._old_policy

        # policy learning
        distribution = policy.pi(training_variables.s_current)
        old_distribution = old_policy.pi(training_variables.s_current)

        self._kl_divergence = NF.mean(old_distribution.kl_divergence(distribution))

        _kl_divergence_grads = nn.grad([self._kl_divergence], policy.get_parameters().values())

        self._kl_divergence_flat_grads = NF.concatenate(*[grad.reshape((-1,)) for grad in _kl_divergence_grads])
        self._kl_divergence_flat_grads.need_grad = True

        log_prob = distribution.log_prob(training_variables.a_current)
        old_log_prob = old_distribution.log_prob(training_variables.a_current)

        prob_ratio = NF.exp(log_prob - old_log_prob)
        advantage = training_variables.extra['advantage']
        self._approximate_return = NF.mean(prob_ratio*advantage)

        _approximate_return_grads = nn.grad([self._approximate_return], policy.get_parameters().values())

        self._approximate_return_flat_grads = NF.concatenate(
            *[grad.reshape((-1,)) for grad in _approximate_return_grads])
        self._approximate_return_flat_grads.need_grad = True

        copy_network_parameters(policy.get_parameters(), old_policy.get_parameters(), tau=1.0)

    def _compute_full_step_params_update(self, policy, s_batch, a_batch, adv_batch, training_variables):
        _, _, approximate_return_flat_grads = self._forward_all_variables(
            s_batch, a_batch, adv_batch, training_variables)

        def fisher_vector_product_wrapper(step_direction):
            return self._fisher_vector_product(policy, s_batch, a_batch,
                                               step_direction, training_variables)

        step_direction = conjugate_gradient(
            fisher_vector_product_wrapper, approximate_return_flat_grads,
            max_iterations=self._config.conjugate_gradient_iterations)

        fisher_vector_product = self._fisher_vector_product(
            policy, s_batch, a_batch, step_direction, training_variables)
        sAs = float(np.dot(step_direction, fisher_vector_product))

        # adding 1e-8 to avoid computational error
        beta = (2.0 * self._config.sigma_kl_divergence_constraint / (sAs + 1e-8)) ** 0.5
        full_step_params_update = beta * step_direction

        return full_step_params_update

    def _fisher_vector_product(self, policy, s_batch, a_batch, vector, training_variables):
        sum_hessian_multiplied_vector = 0
        gpu_batch_size = self._gpu_batch_size(len(s_batch))
        total_blocks = len(s_batch) // gpu_batch_size

        for block_index in range(total_blocks):
            start_idx = block_index * gpu_batch_size
            training_variables.s_current.d = s_batch[start_idx:start_idx+gpu_batch_size]
            training_variables.a_current.d = a_batch[start_idx:start_idx+gpu_batch_size]

            for param in policy.get_parameters().values():
                param.grad.zero()
            self._kl_divergence_flat_grads.forward()
            hessian_vector_product = _hessian_vector_product(self._kl_divergence_flat_grads,
                                                             policy.get_parameters().values(),
                                                             vector)
            hessian_multiplied_vector = hessian_vector_product + self._config.conjugate_gradient_damping * vector
            sum_hessian_multiplied_vector += hessian_multiplied_vector
        return sum_hessian_multiplied_vector / total_blocks

    def _linesearch_and_update_params(
            self, policy, s_batch, a_batch, adv_batch, full_step_params_update, training_variables):
        current_flat_params = _concat_network_params_in_ndarray(policy.get_parameters())

        current_approximate_return, _, _ = self._forward_all_variables(
            s_batch, a_batch, adv_batch, training_variables)

        for step_size in 0.5**np.arange(self._config.maximum_backtrack_numbers):
            new_flat_params = current_flat_params + step_size * full_step_params_update
            _update_network_params_by_flat_params(policy.get_parameters(), new_flat_params)

            approximate_return, kl_divergence, _ = self._forward_all_variables(
                s_batch, a_batch, adv_batch, training_variables)

            improved = approximate_return - current_approximate_return > 0.
            is_in_kl_divergence_constraint = kl_divergence < self._config.sigma_kl_divergence_constraint

            if improved and is_in_kl_divergence_constraint:
                return
            elif not improved:
                logger.debug("TRPO LineSearch: Not improved, Shrink step size and Retry")
            elif not is_in_kl_divergence_constraint:
                logger.debug("TRPO LineSearch: Not fullfill constraints, Shrink step size and Retry")
            else:
                raise RuntimeError("Should not reach here")

        logger.debug("TRPO LineSearch: Reach max iteration so Recover current parmeteres...")
        _update_network_params_by_flat_params(policy.get_parameters(), current_flat_params)

    def _forward_all_variables(self, s_batch, a_batch, adv_batch, training_variables):
        sum_approximate_return = 0.0
        sum_kl_divergence = 0.0
        sum_approximate_return_flat_grad = 0.0
        gpu_batch_size = self._gpu_batch_size(len(s_batch))
        total_blocks = len(s_batch) // gpu_batch_size

        for block_index in range(total_blocks):
            start_idx = block_index * gpu_batch_size
            training_variables.s_current.d = s_batch[start_idx:start_idx+gpu_batch_size]
            training_variables.a_current.d = a_batch[start_idx:start_idx+gpu_batch_size]
            training_variables.extra['advantage'].d = adv_batch[start_idx:start_idx+gpu_batch_size]

            nn.forward_all([self._approximate_return,
                            self._kl_divergence,
                            self._approximate_return_flat_grads])

            sum_approximate_return += float(self._approximate_return.d)
            sum_kl_divergence += float(self._kl_divergence.d)
            sum_approximate_return_flat_grad += self._approximate_return_flat_grads.d

        approximate_return = sum_approximate_return / total_blocks
        approximate_return_flat_grads = sum_approximate_return_flat_grad / total_blocks
        kl_divergence = sum_kl_divergence / total_blocks
        return approximate_return, kl_divergence, approximate_return_flat_grads

    def _setup_training_variables(self, batch_size: int) -> TrainingVariables:
        gpu_batch_size = self._gpu_batch_size(batch_size)
        s_current_var = nn.Variable((gpu_batch_size, *self._env_info.state_shape))
        if self._env_info.is_discrete_action_env():
            a_current_var = nn.Variable((gpu_batch_size, 1))
        else:
            a_current_var = nn.Variable((gpu_batch_size, self._env_info.action_dim))
        advantage_var = nn.Variable((gpu_batch_size, 1))
        extra = {}
        extra['advantage'] = advantage_var
        return TrainingVariables(gpu_batch_size, s_current_var, a_current_var, extra=extra)
