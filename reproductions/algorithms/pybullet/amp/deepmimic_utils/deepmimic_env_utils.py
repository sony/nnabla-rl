# Copyright 2024 Sony Group Corporation.
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

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import gym
import gym.spaces
import numpy as np
from gym import spaces

from nnabla_rl.environments.amp_env import TaskResult
from nnabla_rl.typing import State

import sys  # noqa
try:
    sys.path.append(str(pathlib.Path(__file__).parent.parent / "DeepMimic"))  # noqa
    from DeepMimicCore import DeepMimicCore  # noqa
except ModuleNotFoundError:
    from nnabla_rl.logger import logger
    logger.info("No DeepMimicCore file. Please build the DeepMimic environment and generate the python file.")


def update_core(core: "DeepMimicCore.cDeepMimicCore", update_timesteps: int, agent_id: int
                ) -> Tuple[bool, bool, Dict[str, bool]]:
    num_substeps = core.GetNumUpdateSubsteps()
    timestep = float(update_timesteps) / float(num_substeps)
    num_steps = 0
    for _ in range(num_substeps):
        core.Update(timestep)
        num_steps += 1
        valid_episode = core.CheckValidEpisode()

        if not valid_episode:
            return False, True, {"_valid_episode": valid_episode, "_task_fail": False, "_task_success": False}

        if core.IsEpisodeEnd():
            terminate = core.CheckTerminate(agent_id)
            # 0 is Null, 1 is Fail and 2 is Success
            # See: https://github.com/xbpeng/DeepMimic/blob/70e7c6b22b775bb9342d4e15e6ef0bd91a55c6c0/env/env.py#L7
            return False, True, {"_valid_episode": valid_episode,
                                 "_task_fail": True if terminate == 1 else False,
                                 "_task_success": True if terminate == 2 else False}

        if core.NeedNewAction(agent_id):
            assert num_steps >= num_substeps
            return True, False, {"_valid_episode": valid_episode, "_task_fail": False, "_task_success": False}

    return False, False, {"_valid_episode": valid_episode, "_task_fail": False, "_task_success": False}


def update_core_for_num_substeps(until_action_needed: bool, core: "DeepMimicCore.cDeepMimicCore",
                                 update_timesteps: int, agent_id: int) -> Tuple[bool, Dict[str, Any]]:
    if until_action_needed:
        action_needed = False
        done = False
        while (not action_needed) and (not done):
            action_needed, done, info = update_core(core, update_timesteps, agent_id)
    else:
        action_needed, done, info = update_core(core, update_timesteps, agent_id)

    info["action_needed"] = action_needed
    return done, info


def record_invalid_or_valid_state(num_timesteps: int) -> np.ndarray:
    return np.array([1.0 if num_timesteps > 0.0 else 0.0], dtype=np.float32)


def compile_observation(core: "DeepMimicCore.cDeepMimicCore", num_timesteps: int, agent_id: int) -> State:
    if num_timesteps == 0:
        # In an initial step, a dummy state given as the concatenated st and st+1.
        dummy_state = np.zeros(core.GetAMPObsSize(), dtype=np.float32)
        return (np.array(core.RecordState(agent_id), dtype=np.float32),
                dummy_state,
                record_invalid_or_valid_state(num_timesteps))
    else:
        return (np.array(core.RecordState(agent_id), dtype=np.float32),
                np.array(core.RecordAMPObsAgent(agent_id), dtype=np.float32),
                record_invalid_or_valid_state(num_timesteps))


def compile_goal_env_observation(core: "DeepMimicCore.cDeepMimicCore", num_timesteps: int, agent_id: int
                                 ) -> Dict[str, State]:
    observation = compile_observation(core, num_timesteps, agent_id)
    goal_env_observation = {"observation": observation,
                            "desired_goal": (np.array(core.RecordGoal(agent_id), dtype=np.float32),
                                             np.ones((1,), dtype=np.float32)),
                            # not use an achieved goal
                            "achieved_goal": (np.array(core.RecordGoal(agent_id), dtype=np.float32) * 0.0,
                                              np.zeros((1,), dtype=np.float32))}
    return goal_env_observation


def generate_dummy_goal_env_state(observation_space: gym.spaces.Dict) -> State:
    state: List[np.ndarray] = []
    sample = observation_space.sample()
    for key in ["observation", "desired_goal", "achieved_goal"]:
        s = sample[key]
        if isinstance(s, tuple):
            state.extend(s)
        else:
            state.append(s)
    state = list(map(lambda v: v * 0.0, state))
    assert len(state) == 7
    return tuple(state)


def label_task_result(state, reward, done, info) -> TaskResult:
    task_success = info.pop("_task_success")
    task_fail = info.pop("_task_fail")
    if task_success:
        return TaskResult.SUCCESS
    elif task_fail:
        return TaskResult.FAIL
    else:
        return TaskResult.UNKNOWN


def initialize_env(args_file: str, agent_id: int,
                   enable_window_view: bool = False, seed: Optional[int] = None) -> "DeepMimicCore.cDeepMimicCore":
    core = DeepMimicCore.cDeepMimicCore(enable_window_view)
    if seed is not None:
        core.SeedRand(seed)

    core.ParseArgs(["--arg_file", args_file])
    core.Init()
    core.SetPlaybackSpeed(1)

    # Build all settings
    core.BuildStateNormGroups(agent_id)
    core.BuildStateOffset(agent_id)
    core.BuildStateScale(agent_id)

    if core.GetGoalSize(agent_id) != 0:
        core.BuildGoalNormGroups(agent_id)
        core.BuildGoalOffset(agent_id)
        core.BuildGoalScale(agent_id)

    core.BuildActionOffset(agent_id)
    core.BuildActionScale(agent_id)
    core.BuildActionBoundMax(agent_id)
    core.BuildActionBoundMin(agent_id)
    return core


def load_observation_and_action_space(dummy_core: "DeepMimicCore.cDeepMimicCore",
                                      agent_id: int) -> Tuple[gym.Space, Tuple[np.ndarray, ...], Tuple[np.ndarray, ...],
                                                              gym.Space,  np.ndarray,
                                                              np.ndarray, float, float]:
    action_space = spaces.Box(
        low=np.array(dummy_core.BuildActionBoundMin(agent_id), dtype=np.float32),
        high=np.array(dummy_core.BuildActionBoundMax(agent_id), dtype=np.float32),
        shape=(dummy_core.GetActionSize(agent_id),),
        dtype=np.float32,
    )
    # observation for policy and v function
    # observation for discriminator
    # DeepMimic env returns the concatenated st and st+1.
    # valid or invalid
    observation_space = spaces.Tuple([spaces.Box(low=-np.inf,  # type: ignore
                                                 high=np.inf,
                                                 shape=(dummy_core.GetStateSize(agent_id),),
                                                 dtype=np.float32),
                                      spaces.Box(low=-np.inf,
                                                 high=np.inf,
                                                 shape=(dummy_core.GetAMPObsSize(),),
                                                 dtype=np.float32),
                                      spaces.Box(low=0.0,
                                                 high=1.0,
                                                 shape=(1,),
                                                 dtype=np.float32)])
    # Offset means default + offset, so mean is negative of the offset.
    obs_for_policy_mean = -1.0 * np.array(dummy_core.BuildStateOffset(agent_id), dtype=np.float32)
    obs_for_policy_var = (1.0 / np.array(dummy_core.BuildStateScale(agent_id), dtype=np.float32)) ** 2
    obs_for_reward_mean = -1.0 * np.array(dummy_core.GetAMPObsOffset(), dtype=np.float32)
    obs_for_reward_var = (1.0 / np.array(dummy_core.GetAMPObsScale(), dtype=np.float32)) ** 2
    observation_mean = (obs_for_policy_mean, obs_for_reward_mean, np.zeros((1,), dtype=np.float32))
    observation_var = (obs_for_policy_var, obs_for_reward_var, np.ones((1,), dtype=np.float32))
    action_mean = -1.0 * np.array(dummy_core.BuildActionOffset(agent_id), dtype=np.float32)
    action_var = (1.0 / np.array(dummy_core.BuildActionScale(agent_id), dtype=np.float32)) ** 2
    reward_at_task_fail = float(dummy_core.GetRewardFail(agent_id))
    reward_at_task_success = float(dummy_core.GetRewardSucc(agent_id))
    return (observation_space,
            observation_mean,
            observation_var,
            action_space,
            action_mean,
            action_var,
            reward_at_task_fail,
            reward_at_task_success)


def load_goal_env_observation_and_action_space(dummy_core: "DeepMimicCore.cDeepMimicCore", agent_id: int
                                               ) -> Tuple[gym.Space, Dict[str, Tuple[np.ndarray, ...]],
                                                          Dict[str, Tuple[np.ndarray, ...]],
                                                          gym.Space, np.ndarray, np.ndarray,
                                                          float, float]:
    assert dummy_core.GetGoalSize(agent_id) != 0, "This env does not have a goal! Use DeepMimicEnv."
    (observation_space,
     observation_mean,
     observation_var,
     action_space,
     action_mean,
     action_var,
     reward_at_task_fail,
     reward_at_task_success) = load_observation_and_action_space(dummy_core, agent_id)
    # Add goal state to observation space
    goal_state_space = spaces.Tuple([spaces.Box(low=-np.inf,  # type: ignore[operator]
                                                high=np.inf,
                                                shape=(dummy_core.GetGoalSize(agent_id),),
                                                dtype=np.float32),
                                     # valid or invalid
                                     spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(1,),
                                                dtype=np.float32)])
    goal_env_observation_space = spaces.Dict({"observation": observation_space,
                                              "desired_goal": goal_state_space,
                                              "achieved_goal": goal_state_space})
    # Offset means default + offset, so mean is negative of the offset.
    obs_for_goal_mean = (-1.0 * np.array(dummy_core.BuildGoalOffset(agent_id), dtype=np.float32),
                         np.zeros((1,), dtype=np.float32))
    obs_for_goal_var = ((1.0 / np.array(dummy_core.BuildGoalScale(agent_id), dtype=np.float32)) ** 2,
                        np.ones((1,), dtype=np.float32))

    goal_env_observation_mean = {"observation": observation_mean,
                                 "desired_goal": obs_for_goal_mean,
                                 "achieved_goal": obs_for_goal_mean}
    goal_env_observation_var = {"observation": observation_var,
                                "desired_goal": obs_for_goal_var,
                                "achieved_goal": obs_for_goal_var}
    return (goal_env_observation_space,
            goal_env_observation_mean,
            goal_env_observation_var,
            action_space,
            action_mean,
            action_var,
            reward_at_task_fail,
            reward_at_task_success)
