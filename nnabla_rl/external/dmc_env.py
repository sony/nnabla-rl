# This file is a fork from a MIT-licensed project named rl_with_resets:
# https://github.com/evgenii-nikishin/rl_with_resets/blob/main/continuous_control/wrappers/dmc_env.py (#502ec52)
# The source code is provided under the licenses described below.

# The MIT License
#
# Copyright (c) 2022 Evgenii Nikishin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Taken from
# https://github.com/denisyarats/dmc2gym
# and modified to exclude duplicated code.

# The MIT License
#
# Copyright (c) 2020 Denis Yarats
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
from typing import Dict, Optional, OrderedDict

import numpy as np
from dm_control import suite
from dm_env import specs
from gym import core, spaces
from gym.envs.registration import EnvSpec


def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, specs.BoundedArray):
        return spaces.Box(low=spec.minimum,
                          high=spec.maximum,
                          shape=spec.shape,
                          dtype=spec.dtype)
    elif isinstance(spec, specs.Array):
        return spaces.Box(low=-float('inf'),
                          high=float('inf'),
                          shape=spec.shape,
                          dtype=spec.dtype)
    else:
        raise NotImplementedError


class DMCEnv(core.Env):
    def __init__(self,
                 domain_name: str,
                 task_name: str,
                 task_kwargs: Optional[Dict] = {},
                 environment_kwargs=None):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'  # type: ignore

        self._env = suite.load(domain_name=domain_name,
                               task_name=task_name,
                               task_kwargs=task_kwargs,
                               environment_kwargs=environment_kwargs)

        self.spec = EnvSpec(f'{domain_name}-{task_name}-v0')
        self.action_space = dmc_spec2gym_space(self._env.action_spec())
        self.observation_space = dmc_spec2gym_space(
            self._env.observation_spec())

        self.seed(seed=task_kwargs['random'])  # type: ignore

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        obs = time_step.observation

        info = {}
        if done and time_step.discount == 1.0:
            info['TimeLimit.truncated'] = True

        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return time_step.observation

    def render(self,
               mode='rgb_array',
               height: int = 84,
               width: int = 84,
               camera_id: int = 0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self._env.physics.render(height=height,
                                        width=width,
                                        camera_id=camera_id)
