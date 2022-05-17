# Copyright 2022 Sony Group Corporation.
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

from nnabla.solver import Solver


class SolverWrapper(Solver):
    def __init__(self, solver):
        super().__init__()
        self._solver = solver

    def setup(self, params):
        self._solver.setup(params)

    def set_parameters(self, param_dict, reset=True, retain_state=False):
        self._solver.set_parameters(param_dict, reset=reset, retain_state=retain_state)

    def remove_parameters(self, keys):
        self._solver.remove_parameters(keys)

    def clear_parameters(self):
        self._solver.clear_parmeters()

    def get_parmaters(self):
        return self._solver.get_parameters()

    def get_states(self):
        return self._solver.get_states()

    def set_states(self, states):
        return self._solver.set_states()

    def save_states(self, path):
        self._solver.save_states(path)

    def set_states_to_protobuf(self, optimizer):
        self._solver.set_states_to_protobuf(optimizer)

    def load_states(self, path):
        self._solver.load_states(path)

    def set_states_from_protobuf(self, optimizer_proto):
        self._solver.set_states_from_protobuf(optimizer_proto)

    def set_learning_rate(self, learning_rate):
        self._solver.set_learning_rate(learning_rate)

    def zero_grad(self):
        self._solver.zero_grad()

    def update(self, update_pre_hook=None, update_post_hook=None):
        self._solver.update(update_pre_hook, update_post_hook)

    def weight_decay(self, decay_rate, pre_hook=None, post_hook=None):
        self._solver.weight_decay(decay_rate, pre_hook, post_hook)

    def clip_grad_by_norm(self, clip_norm, pre_hook=None, post_hook=None):
        self._solver.clip_grad_by_norm(clip_norm, pre_hook, post_hook)

    def check_inf_grad(self, pre_hook=None, post_hook=None):
        return self._solver.check_inf_grad(pre_hook, post_hook)

    def check_nan_grad(self, pre_hook=None, post_hook=None):
        return self._solver.check_nan_grad(pre_hook, post_hook)

    def check_inf_or_nan_grad(self, pre_hook=None, post_hook=None):
        return self._solver.check_inf_or_nan_grad(pre_hook, post_hook)

    def scale_grad(self, scale, pre_hook=None, post_hook=None):
        self._solver.check_scale_grad(scale, pre_hook, post_hook)

    @property
    def name(self):
        """
        Get the name of the solver.
        """
        return self._solver.name

    def learning_rate(self):
        return self._solver.learning_rate()


class UpdateWrapper(SolverWrapper):
    def update(self, update_pre_hook=None, update_post_hook=None):
        self.before_update()
        super().update(update_pre_hook, update_post_hook)
        self.after_update()

    def before_update(self):
        pass

    def after_update(self):
        pass


class AutoWeightDecay(UpdateWrapper):
    def __init__(self, solver: Solver, decay_rate: float):
        super().__init__(solver)
        self._decay_rate = decay_rate

    def before_update(self):
        self.weight_decay(self._decay_rate)


class AutoClipGradByNorm(UpdateWrapper):
    def __init__(self, solver: Solver, clip_norm: float):
        super().__init__(solver)
        self._clip_norm = clip_norm

    def before_update(self):
        self.clip_grad_by_norm(self._clip_norm)
