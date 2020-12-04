from typing import Optional, Iterable, Dict

import numpy as np

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

from nnabla_rl.model_trainers.model_trainer import TrainerParam, Training, TrainingVariables, ModelTrainer
from nnabla_rl.models import Model, StochasticPolicy
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.utils.optimization import conjugate_gradient
from nnabla_rl.logger import logger


def _hessian_vector_product(flat_grads, params, vector):
    """ Compute multiplied vector hessian of parameters and vector

    Args:
        flat_grads (nn.Variable): gradient of parameters, should be flattened
        params (list[nn.Variable]): list of parameters
        vector (numpy.ndarray): input vector, shape is the same as flat_grads
    Returns:
        hessian_vector (numpy.ndarray): multiplied vector of hessian of parameters and vector
    See:
        https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/
    """
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
    """ Concatenate network parameters in numpy.ndarray,
        this function returns copied parameters

    Args:
        params (OrderedDict): parameters
    Returns:
        flat_params (numpy.ndarray): flatten parameters in numpy.ndarray type
    """
    flat_params = []
    for param in params.values():
        flat_param = param.d.copy().flatten()
        flat_params.append(flat_param)
    return np.concatenate(flat_params)


def _update_network_params_by_flat_params(params, new_flat_params):
    """ Update Network parameters by hand

    Args:
        params (OrderedDict): parameteres
        new_flat_params (numpy.ndarray): flattened new parameters
    """
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
class TRPOTrainingVariables(TrainingVariables):
    advantage: Optional[nn.Variable] = None


@dataclass
class TRPOPolicyTrainerParam(TrainerParam):
    batch_size: int = 5000
    num_steps_per_iteration: int = 5000
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.1
    conjugate_gradient_iterations: int = 20

    def __post_init__(self):
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.num_steps_per_iteration, 'num_steps_per_iteration')
        self._assert_positive(self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(self.conjugate_gradient_damping, 'conjugate_gradient_damping')
        self._assert_positive(self.conjugate_gradient_iterations, 'conjugate_gradient_iterations')


class TRPOPolicyTrainer(ModelTrainer):
    def __init__(self, env_info, params=TRPOPolicyTrainerParam()):
        super(TRPOPolicyTrainer, self).__init__(env_info, params)
        self._old_policy = None

        self._approximate_return = None
        self._approximate_return_flat_grads = None
        self._kl_divergence = None
        self._kl_divergence_flat_grads = None

        self._train_count = 0

    # Override train to avoid creating training variables every iteration
    def train(self, experience, **kwargs) -> Dict:
        if self._models is None:
            raise RuntimeError('Call setup_training() first. Model is not set!')
        self._train_count += 1
        self._training.before_update(self._train_count)
        error_info = self._update_model(self._models, self._solvers, experience, self._training_variables, **kwargs)
        self._training.after_update(self._train_count)

        return error_info

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      experience,
                      training_variables: TrainingVariables,
                      **kwargs):
        (s, a, advantage, * _) = experience

        policy = models[0]
        old_policy = self._old_policy

        full_step_params_update = self._compute_full_step_params_update(policy, s, a, advantage, training_variables)

        self._linesearch_and_update_params(policy, s, a, advantage, full_step_params_update, training_variables)

        copy_network_parameters(policy.get_parameters(), old_policy.get_parameters(), tau=1.0)

        errors = {}
        return errors

    def _build_training_graph(self, models: Iterable[Model],
                              training: Training,
                              training_variables: TrainingVariables):
        if len(models) != 1:
            raise RuntimeError('TRPO training only support 1 model for training')
        if not isinstance(models[0], StochasticPolicy):
            raise ValueError
        policy = models[0]
        if self._old_policy is None:
            self._old_policy = policy.deepcopy('old_policy')
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
        self._approximate_return = NF.mean(prob_ratio*training_variables.advantage)

        _approximate_return_grads = nn.grad([self._approximate_return], policy.get_parameters().values())

        self._approximate_return_flat_grads = NF.concatenate(
            *[grad.reshape((-1,)) for grad in _approximate_return_grads])
        self._approximate_return_flat_grads.need_grad = True

        copy_network_parameters(policy.get_parameters(), old_policy.get_parameters(), tau=1.0)

    def _compute_full_step_params_update(self, policy, s_batch, a_batch, adv_batch, training_variables):
        _, _, approximate_return_flat_grads = self._forward_all_variables(
            s_batch, a_batch, adv_batch, training_variables)

        def fisher_vector_product_wrapper(step_direction):
            return self._fisher_vector_product(policy, s_batch, a_batch, step_direction, training_variables)

        step_direction = conjugate_gradient(
            fisher_vector_product_wrapper, approximate_return_flat_grads,
            max_iterations=self._params.conjugate_gradient_iterations)

        fisher_vector_product = self._fisher_vector_product(
            policy, s_batch, a_batch, step_direction, training_variables)
        sAs = float(np.dot(step_direction, fisher_vector_product))

        # adding 1e-8 to avoid computational error
        beta = (2.0 * self._params.sigma_kl_divergence_constraint / (sAs + 1e-8)) ** 0.5
        full_step_params_update = beta * step_direction

        return full_step_params_update

    def _fisher_vector_product(self, policy, s_batch, a_batch, vector, training_variables):
        sum_hessian_multiplied_vector = 0

        batch_size = self._params.batch_size
        total_blocks = self._params.num_steps_per_iteration // batch_size

        for block_index in range(total_blocks):
            start_idx = block_index * batch_size
            training_variables.s_current.d = s_batch[start_idx:start_idx+batch_size]
            training_variables.a_current.d = a_batch[start_idx:start_idx+batch_size]

            for param in policy.get_parameters().values():
                param.grad.zero()
            self._kl_divergence_flat_grads.forward()
            hessian_vector_product = _hessian_vector_product(self._kl_divergence_flat_grads,
                                                             policy.get_parameters().values(),
                                                             vector)
            hessian_multiplied_vector = hessian_vector_product + self._params.conjugate_gradient_damping * vector
            sum_hessian_multiplied_vector += hessian_multiplied_vector
        return sum_hessian_multiplied_vector / total_blocks

    def _linesearch_and_update_params(
            self, policy, s_batch, a_batch, adv_batch, full_step_params_update, training_variables):
        current_flat_params = _concat_network_params_in_ndarray(policy.get_parameters())

        current_approximate_return, _, _ = self._forward_all_variables(s_batch, a_batch, adv_batch, training_variables)

        for step_size in 0.5**np.arange(self._params.maximum_backtrack_numbers):
            new_flat_params = current_flat_params + step_size * full_step_params_update
            _update_network_params_by_flat_params(policy.get_parameters(), new_flat_params)

            approximate_return, kl_divergence, _ = self._forward_all_variables(
                s_batch, a_batch, adv_batch, training_variables)

            improved = approximate_return - current_approximate_return > 0.
            is_in_kl_divergence_constraint = kl_divergence < self._params.sigma_kl_divergence_constraint

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
        batch_size = self._params.batch_size
        total_blocks = self._params.num_steps_per_iteration // batch_size

        sum_approximate_return = 0.0
        sum_kl_divergence = 0.0
        sum_approximate_return_flat_grad = 0.0

        for block_index in range(total_blocks):
            start_idx = block_index * batch_size
            training_variables.s_current.d = s_batch[start_idx:start_idx+batch_size]
            training_variables.a_current.d = a_batch[start_idx:start_idx+batch_size]
            training_variables.advantage.d = adv_batch[start_idx:start_idx+batch_size]

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
        # Training input variables
        # Use batch_size specied by the user
        batch_size = self._params.batch_size
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        if self._env_info.is_discrete_action_env():
            a_current_var = nn.Variable((batch_size, 1))
        else:
            a_current_var = nn.Variable((batch_size, self._env_info.action_dim))
        advantage_var = nn.Variable((batch_size, 1))
        return TRPOTrainingVariables(s_current_var, a_current_var, advantage=advantage_var)
