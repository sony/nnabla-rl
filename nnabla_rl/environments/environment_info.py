from dataclasses import dataclass, asdict

import gym


@dataclass
class EnvironmentInfo(object):
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    max_episode_steps: int

    @staticmethod
    def from_env(env):
        return EnvironmentInfo(observation_space=env.observation_space,
                               action_space=env.action_space,
                               max_episode_steps=EnvironmentInfo._extract_max_episode_steps(env))

    def is_discrete_action_env(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    @staticmethod
    def _extract_max_episode_steps(env):
        if env.spec is None or env.spec.max_episode_steps is None:
            return float("inf")
        else:
            return env.spec.max_episode_steps
