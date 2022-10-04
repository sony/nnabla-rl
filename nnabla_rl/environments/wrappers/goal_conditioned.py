# Copyright 2021,2022 Sony Group Corporation.
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

from typing import Optional, SupportsFloat, Tuple, cast

import gym
from gym import spaces

from nnabla_rl.external.goal_env import GoalEnv


class GoalEnvWrapper(GoalEnv):
    def __init__(self, env: gym.Env) -> None:
        """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.
        Args:
            env: The environment to wrap
        """
        self._gym_env = env
        self._action_space: Optional[spaces.Space] = None
        self._observation_space: Optional[spaces.Space] = None
        self._reward_range: Optional[Tuple[SupportsFloat, SupportsFloat]] = None
        self._metadata: Optional[dict] = None

    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self._gym_env, name)

    @property
    def spec(self):
        """Returns the environment specification."""
        return self._gym_env.spec

    @classmethod
    def class_name(cls):
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property  # type: ignore
    def action_space(self) -> spaces.Space:
        """Returns the action space of the environment."""
        if self._action_space is None:
            return self._gym_env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space):
        self._action_space = space

    @property  # type: ignore
    def observation_space(self) -> spaces.Space:
        """Returns the observation space of the environment."""
        if self._observation_space is None:
            return self._gym_env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space):
        self._observation_space = space

    @property  # type: ignore
    def reward_range(self) -> Tuple[SupportsFloat, SupportsFloat]:
        """Return the reward range of the environment."""
        if self._reward_range is None:
            reward_range: Tuple[SupportsFloat, SupportsFloat] = self._gym_env.reward_range
            return reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: Tuple[SupportsFloat, SupportsFloat]):
        self._reward_range = value

    @property  # type: ignore
    def metadata(self) -> dict:
        """Returns the environment metadata."""
        if self._metadata is None:
            metadata: dict = self._gym_env.metadata
            return metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def np_random(self):
        """Returns the environment np_random."""
        return self._gym_env.np_random

    @np_random.setter
    def np_random(self, value):
        self._gym_env.np_random = value

    @property
    def _np_random(self):
        raise AttributeError(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        )

    def step(self, action):
        """Steps through the environment with action."""
        return self._gym_env.step(action)

    def reset(self, **kwargs):
        """Resets the environment with kwargs."""
        return self._gym_env.reset(**kwargs)

    def render(self, **kwargs):
        """Renders the environment with kwargs."""
        return self._gym_env.render(**kwargs)

    def close(self):
        """Closes the environment."""
        return self._gym_env.close()

    def seed(self, seed=None):
        """Seeds the environment."""
        return self._gym_env.seed(seed)

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self._gym_env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._gym_env.compute_reward(achieved_goal, desired_goal, info)


class GoalConditionedTupleObservationEnv(gym.ObservationWrapper):
    def __init__(self, env: GoalEnv):
        super(GoalConditionedTupleObservationEnv, self).__init__(env)
        self._observation_keys = ['observation', 'desired_goal', 'achieved_goal']

        self._check_env(env)
        self._observation_space = self._build_observation_space(env)

    def _check_env(self, env: GoalEnv):
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("GoalEnv requires an observation space of type gym.spaces.Dict")

        for key in env.observation_space.spaces:
            if key not in self._observation_keys:
                error_msg = f'{key} should be included in observation_space!!'
                raise ValueError(error_msg)

    def _build_observation_space(self, env: GoalEnv):
        observation_spaces = []
        raw_observation_spaces = cast(gym.spaces.Dict, env.observation_space)
        for key in self._observation_keys:
            observation_space = raw_observation_spaces[key]
            observation_spaces.append(observation_space)
        return gym.spaces.Tuple(observation_spaces)  # type: ignore

    def observation(self, observation):
        self._check_observation(observation)
        return tuple(observation[key].copy() for key in self._observation_keys)

    def _check_observation(self, observation):
        for key in observation.keys():
            if key not in self._observation_keys:
                error_msg = f'{key} should be included in observations!!'
                raise ValueError(error_msg)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info=info)
