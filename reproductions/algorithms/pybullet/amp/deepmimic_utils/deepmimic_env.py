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
from typing import Callable, Optional, cast

import gym
import numpy as np
from gym.envs.registration import EnvSpec

from nnabla_rl.environments.amp_env import AMPEnv, AMPGoalEnv, TaskResult
from nnabla_rl.typing import Experience

import sys  # noqa
sys.path.append(str(pathlib.Path(__file__).parent))  # noqa
from deepmimic_env_utils import (update_core_for_num_substeps, initialize_env, compile_observation,  # noqa
                                 generate_dummy_goal_env_state, compile_goal_env_observation,
                                 load_goal_env_observation_and_action_space, load_observation_and_action_space,
                                 record_invalid_or_valid_state, label_task_result)

try:
    sys.path.append(str(pathlib.Path(__file__).parent.parent / "DeepMimic"))  # noqa
    from DeepMimicCore import DeepMimicCore  # noqa
except ModuleNotFoundError:
    from nnabla_rl.logger import logger
    logger.info("No DeepMimicCore file. Please build the DeepMimic environment and generate the python file.")

try:
    from OpenGL.GLUT import (GLUT_DEPTH, GLUT_DOUBLE, GLUT_ELAPSED_TIME, GLUT_RGBA, glutCreateWindow, glutDisplayFunc,
                             glutGet, glutInit, glutInitDisplayMode, glutInitWindowSize, glutKeyboardFunc,
                             glutLeaveMainLoop, glutMainLoop, glutMotionFunc, glutMouseFunc, glutPostRedisplay,
                             glutReshapeFunc, glutSwapBuffers, glutTimerFunc)
except ModuleNotFoundError:
    from nnabla_rl.logger import logger
    logger.info("No OpenGL lib. Please build the DeepMimic environment and generate the python file, "
                "OpenGL lib is installed automatically.")


class DeepMimicEnv(AMPEnv):
    unwrapped: "DeepMimicEnv"

    def __init__(self, args_file: str, eval_mode: bool,
                 fps: int = 60, num_processes: int = 1, step_until_action_needed: bool = True) -> None:
        assert fps > 0
        assert num_processes > 0

        self._args_file = args_file
        self._eval_mode = eval_mode
        self._step_until_action_needed = step_until_action_needed
        self._num_processes = num_processes
        self._update_timesteps = 1.0 / float(fps)
        self._agent_id = 0  # NOTE: agent id is always 0.
        self._total_valid_step_count = 0
        self._seed = None
        self._initialized = False
        self._action_needed = True

        (self.reward_range,
         self.observation_space,
         self.observation_mean,
         self.observation_var,
         self.action_space,
         self.action_mean,
         self.action_var,
         self.reward_at_task_fail,
         self.reward_at_task_success) = self._reward_range_state_and_action_space()
        self.spec = EnvSpec(pathlib.Path(args_file).name.replace('train_amp_', '').replace('_args.txt', '-v0'))

        super().__init__()

    def reset(self):
        if not self._initialized:
            self._core = initialize_env(self._args_file, self._agent_id, seed=self._seed)
            self._total_valid_step_count = 0
            self._initialized = True

        self._num_timesteps = 0
        self._action_needed = True
        self._core.Reset()
        # 0 means TRAIN and 1 means TEST
        # See: https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/scenes/RLScene.h#L11
        if self._eval_mode:
            self._core.SetMode(1)
        else:
            self._core.SetMode(0)

        assert self._core.NeedNewAction(self._agent_id)
        return compile_observation(self._core, self._num_timesteps, self._agent_id)

    def render(self):
        raise NotImplementedError("Use DeepMimicEnvViewer instead of calling render().")

    def seed(self, seed=None):
        self._seed = seed
        super().seed(seed)
        # NOTE: Need to regenerate env to apply the random seed.
        self._core = initialize_env(self._args_file, self._agent_id, seed=self._seed)
        return [seed]

    def task_result(self, state, reward, done, info) -> TaskResult:
        return cast(TaskResult, label_task_result(state, reward, done, info))

    def is_valid_episode(self, state, reward, done, info) -> bool:
        valid_episode = info.pop("_valid_episode")
        return cast(bool, valid_episode)

    def expert_experience(self, state, reward, done, info) -> Experience:
        amp_expert_state = np.array(self._core.RecordAMPObsExpert(self._agent_id), dtype=np.float32)
        expert_state = list(map(lambda v: v * 0.0, self.observation_space.sample()))
        assert expert_state[1].shape == amp_expert_state.shape
        expert_state[1] = amp_expert_state.copy()
        expert_state[2] = record_invalid_or_valid_state(self._num_timesteps)
        dummy_action = 0.0 * self.action_space.sample()
        dummy_next_state = list(map(lambda v: v * 0.0, self.observation_space.sample()))
        return tuple(expert_state), dummy_action, 0.0, False, tuple(dummy_next_state), {}

    def update_sample_counts(self):
        # NOTE: In the deepmimic environment, the sample counts should be the total number
        # of valid steps across all processes.
        # Instead of obtaining the number of valid steps from other processes and summing them,
        # multiply self._num_processes and estimate the approximate value.
        self._core.SetSampleCount(self._num_processes * self._total_valid_step_count)

    def _step(self, action):
        self._num_timesteps += 1
        if self._action_needed:
            self._core.SetAction(self._agent_id, np.array(action, dtype=np.float32).tolist())

        done, info = update_core_for_num_substeps(until_action_needed=self._step_until_action_needed,
                                                  core=self._core, update_timesteps=self._update_timesteps,
                                                  agent_id=self._agent_id)
        self._action_needed = info["action_needed"]
        next_state = compile_observation(self._core, self._num_timesteps, self._agent_id)

        reward = self._core.CalcReward(self._agent_id)

        if done and info["_valid_episode"]:
            self._total_valid_step_count += self._num_timesteps

        return next_state, reward, done, info

    def _reward_range_state_and_action_space(self):
        dummy_core = DeepMimicCore.cDeepMimicCore(False)
        dummy_core.ParseArgs(["--arg_file", self._args_file])
        dummy_core.Init()
        assert dummy_core.GetGoalSize(self._agent_id) == 0, "This env has a goal! Use DeepMimicGoalEnv."

        (observation_space,
         observation_mean,
         observation_var,
         action_space,
         action_mean,
         action_var,
         reward_at_task_fail,
         reward_at_task_success) = load_observation_and_action_space(dummy_core, self._agent_id)
        reward_range = (dummy_core.GetRewardMin(self._agent_id), dummy_core.GetRewardMax(self._agent_id))

        return (reward_range,
                observation_space,
                observation_mean,
                observation_var,
                action_space,
                action_mean,
                action_var,
                reward_at_task_fail,
                reward_at_task_success)


class DeepMimicGoalEnv(AMPGoalEnv):
    unwrapped: "DeepMimicGoalEnv"

    def __init__(self, args_file: str, eval_mode: bool,
                 fps: int = 60, num_processes: int = 1, step_until_action_needed: bool = True) -> None:
        assert fps > 0
        assert num_processes > 0

        self._args_file = args_file
        self._eval_mode = eval_mode
        self._step_until_action_needed = step_until_action_needed
        self._num_processes = num_processes
        self._update_timesteps = 1.0 / float(fps)
        self._agent_id = 0  # NOTE: agent id is always 0.
        self._total_valid_step_count = 0
        self._seed = None
        self._initialized = False
        self._action_needed = True

        (self.reward_range,
         self.observation_space,
         self.observation_mean,
         self.observation_var,
         self.action_space,
         self.action_mean,
         self.action_var,
         self.reward_at_task_fail,
         self.reward_at_task_success) = self._reward_range_state_and_action_space()
        assert self.reward_at_task_fail < self.reward_at_task_success
        self.spec = EnvSpec(pathlib.Path(args_file).name.replace('train_amp_', '').replace('_args.txt', '-v0'))

        super().__init__()

    def reset(self):
        if not self._initialized:
            self._core = initialize_env(self._args_file, self._agent_id, seed=self._seed)
            assert self._core.EnableAMPTaskReward()
            self._total_valid_step_count = 0
            self._initialized = True

        self._num_timesteps = 0
        self._action_needed = True
        self._core.Reset()
        # 0 means TRAIN and 1 means TEST
        # See: https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/scenes/RLScene.h#L11
        if self._eval_mode:
            self._core.SetMode(1)
        else:
            self._core.SetMode(0)

        assert self._core.NeedNewAction(self._agent_id)
        return compile_goal_env_observation(self._core, self._num_timesteps, self._agent_id)

    def render(self):
        raise NotImplementedError("Use DeepMimicEnvViewer instead of calling render().")

    def seed(self, seed=None):
        self._seed = seed
        super().seed(seed)
        # Need to regenerate env to apply the random seed.
        self._core = initialize_env(self._args_file, self._agent_id, seed=self._seed)
        self._initialized = True
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        # NOTE: In deepmimic env, reward is computed intenally.
        return self._core.CalcReward(self._agent_id)

    def task_result(self, state, reward, done, info) -> TaskResult:
        return cast(TaskResult, label_task_result(state, reward, done, info))

    def is_valid_episode(self, state, reward, done, info) -> bool:
        valid_episode = info.pop("_valid_episode")
        return cast(bool, valid_episode)

    def expert_experience(self, state, reward, done, info) -> Experience:
        expert_state_and_next_state = np.array(self._core.RecordAMPObsExpert(self._agent_id), dtype=np.float32)
        state = list(generate_dummy_goal_env_state(self.observation_space))
        assert state[1].shape == expert_state_and_next_state.shape
        state[1] = expert_state_and_next_state.copy()
        state[2] = record_invalid_or_valid_state(self._num_timesteps)
        dummy_action = 0.0 * self.action_space.sample()
        dummy_next_state = generate_dummy_goal_env_state(self.observation_space)
        return tuple(state), dummy_action, 0.0, False, tuple(dummy_next_state), {}

    def update_sample_counts(self):
        # NOTE: In the deepmimic goal environment, the sample counts should be the total number
        # of valid steps across all processes.
        # Instead of obtaining the number of valid steps from other processes and summing them,
        # multiply self._num_processes and estimate the approximate value.
        self._core.SetSampleCount(self._num_processes * self._total_valid_step_count)

    def _step(self, action):
        self._num_timesteps += 1
        if self._action_needed:
            self._core.SetAction(self._agent_id, np.array(action, dtype=np.float32).tolist())

        done, info = update_core_for_num_substeps(until_action_needed=self._step_until_action_needed,
                                                  core=self._core, update_timesteps=self._update_timesteps,
                                                  agent_id=self._agent_id)
        self._action_needed = info["action_needed"]
        next_state = compile_goal_env_observation(self._core, self._num_timesteps, self._agent_id)

        reward = self.compute_reward(next_state["achieved_goal"], next_state["desired_goal"], info)

        if done and info["_valid_episode"]:
            self._total_valid_step_count += self._num_timesteps

        return next_state, reward, done, info

    def _reward_range_state_and_action_space(self):
        dummy_core = DeepMimicCore.cDeepMimicCore(False)
        dummy_core.ParseArgs(["--arg_file", self._args_file])
        dummy_core.Init()
        assert dummy_core.GetGoalSize(self._agent_id) != 0, "This env does not have a goal! Use DeepMimicEnv."

        (observation_space,
         observation_mean,
         observation_var,
         action_space,
         action_mean,
         action_var,
         reward_at_task_fail,
         reward_at_task_success) = load_goal_env_observation_and_action_space(dummy_core, self._agent_id)
        reward_range = (dummy_core.GetRewardMin(self._agent_id), dummy_core.GetRewardMax(self._agent_id))

        return (reward_range,
                observation_space,
                observation_mean,
                observation_var,
                action_space,
                action_mean,
                action_var,
                reward_at_task_fail,
                reward_at_task_success)


class DeepMimicWindowViewer:
    def __init__(self,
                 env: gym.Env,
                 policy_callback_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                 width: int = 800,
                 height: int = 450,
                 playback_speed: int = 1) -> None:
        self._env = env
        assert isinstance(env.unwrapped, DeepMimicEnv) or isinstance(env.unwrapped, DeepMimicGoalEnv)
        self._env_unwrapped = env.unwrapped
        self._width = width
        self._height = height
        self._update_timesteps = self._env_unwrapped._update_timesteps
        self._display_anim_time = int(1000.0 * self._env_unwrapped._update_timesteps)
        self._playback_speed = playback_speed
        self._reshaping = False
        self._action_needed = True
        self._initial_step = True
        self._prev_time = 0.0
        self._updates_per_sec = 0
        self._current_episodes = 0
        self._policy_callback_function = policy_callback_function

        # NOTE: Create window first, then rendering should be enabled.
        self._initialize_window()
        self._core = initialize_env(self._env_unwrapped._args_file, self._env_unwrapped._agent_id,
                                    seed=self._env_unwrapped._seed, enable_window_view=True)
        self._env_unwrapped._core = self._core  # Force to overwrite core
        self._setup_draw()

    def render(self, num_episodes: int):
        self._prev_time = glutGet(GLUT_ELAPSED_TIME)
        self._num_episodes = num_episodes
        glutMainLoop()

    def _initialize_window(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self._width, self._height)
        glutCreateWindow(b"DeepMimic")

    def _setup_draw(self):
        glutDisplayFunc(self._draw)
        glutReshapeFunc(self._reshape)
        glutKeyboardFunc(self._keyboard)
        glutMouseFunc(self._mouse_click)
        glutMotionFunc(self._mouse_move)
        glutTimerFunc(self._display_anim_time, self._animate, 0)

        self._reshape(self._width, self._height)
        self._core.Reshape(self._width, self._height)

    def _draw(self):
        self._update_intermediate_buffer()
        self._core.Draw()
        glutSwapBuffers()
        self._reshaping = False

    def _reshape(self, width, height):
        self._width = width
        self._height = height
        self._reshaping = True

        glutPostRedisplay()

    def _mouse_click(self, button, state, x, y):
        self._core.MouseClick(button, state, x, y)
        glutPostRedisplay()

    def _mouse_move(self, x, y):
        self._core.MouseMove(x, y)
        glutPostRedisplay()

    def _keyboard(self, key, x, y):
        key_val = int.from_bytes(key, byteorder="big")
        self._core.Keyboard(key_val, x, y)

        if (key == b"r"):
            self._state = self._env.reset()
            self._initial_step = False

        glutPostRedisplay()

    def _update_intermediate_buffer(self):
        if not (self._reshaping):
            if self._width != self._core.GetWinWidth() or self._height != self._core.GetWinHeight():
                self._core.Reshape(self._width, self._height)

    def _animate(self, callback_val):
        counter_decay = 0

        num_steps = 1
        curr_time = glutGet(GLUT_ELAPSED_TIME)
        time_elapsed = curr_time - self._prev_time
        self._prev_time = curr_time

        done = self._update_env()

        update_count = num_steps / (0.001 * time_elapsed)
        if np.isfinite(update_count):
            self._updates_per_sec = counter_decay * self._updates_per_sec + (1 - counter_decay) * update_count
            self._core.SetUpdatesPerSec(self._updates_per_sec)

        timer_step = self._calc_display_anim_time(num_steps)
        update_dur = glutGet(GLUT_ELAPSED_TIME) - curr_time
        timer_step -= update_dur
        timer_step = np.maximum(timer_step, 0)

        glutTimerFunc(int(timer_step), self._animate, 0)
        glutPostRedisplay()

        if done:
            self._initial_step = True
            self._shutdown()
            self._current_episodes += 1
            if self._current_episodes >= self._num_episodes:
                self._current_episodes = 0
                glutLeaveMainLoop()

    def _shutdown(self):
        self._core.Shutdown()

    def _calc_display_anim_time(self, num_timestes):
        anim_time = int(self._display_anim_time * num_timestes / self._playback_speed)
        anim_time = np.abs(anim_time)
        return anim_time

    def _update_env(self):
        if self._initial_step:
            self._state = self._env.reset()
            self._initial_step = False

        if self._action_needed:
            self._action = self._policy_callback_function(self._state)

        self._state, reward, done, info = self._env.step(self._action)
        self._action_needed = info["action_needed"]
        return done
