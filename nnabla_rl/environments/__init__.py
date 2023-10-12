# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023 Sony Group Corporation.
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

from gym.envs.registration import register

from nnabla_rl.environments.dummy import (DummyAtariEnv, DummyContinuous, DummyContinuousActionGoalEnv, DummyDiscrete,  # noqa
                                          DummyDiscreteActionGoalEnv, DummyDiscreteImg, DummyContinuousImg,
                                          DummyFactoredContinuous, DummyMujocoEnv,
                                          DummyTupleContinuous, DummyTupleDiscrete, DummyTupleMixed,
                                          DummyTupleStateContinuous, DummyTupleStateDiscrete,
                                          DummyTupleActionContinuous, DummyTupleActionDiscrete)

register(
    id='FakeMujocoNNablaRL-v1',
    entry_point='nnabla_rl.environments.dummy:DummyMujocoEnv',
    max_episode_steps=10
)

register(
    id='FakeAtariNNablaRLNoFrameskip-v1',
    entry_point='nnabla_rl.environments.dummy:DummyAtariEnv',
    max_episode_steps=10
)

register(
    id='FakeGoalConditionedNNablaRL-v1',
    entry_point='nnabla_rl.environments.dummy:DummyContinuousActionGoalEnv',
    max_episode_steps=10
)

register(
    id='FactoredLunarLanderContinuousV2NNablaRL-v1',
    entry_point='nnabla_rl.environments.factored_envs:FactoredLunarLanderV2',
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200.0,
)

register(
    id='FactoredAntV4NNablaRL-v1',
    entry_point='nnabla_rl.environments.factored_envs:FactoredAntV4',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='FactoredHopperV4NNablaRL-v1',
    entry_point='nnabla_rl.environments.factored_envs:FactoredHopperV4',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='FactoredHalfCheetahV4NNablaRL-v1',
    entry_point='nnabla_rl.environments.factored_envs:FactoredHalfCheetahV4',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='FactoredWalker2dV4NNablaRL-v1',
    entry_point='nnabla_rl.environments.factored_envs:FactoredWalker2dV4',
    max_episode_steps=1000,
)

register(
    id='FactoredHumanoidV4NNablaRL-v1',
    entry_point='nnabla_rl.environments.factored_envs:FactoredHumanoidV4',
    max_episode_steps=1000,
)
