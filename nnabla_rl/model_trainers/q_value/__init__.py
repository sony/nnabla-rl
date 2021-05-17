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

from nnabla_rl.model_trainers.q_value.bcq_q_trainer import (  # noqa
    BCQQTrainer, BCQQTrainerConfig)
from nnabla_rl.model_trainers.q_value.categorical_dqn_q_trainer import (  # noqa
    CategoricalDQNQTrainer, CategoricalDQNQTrainerConfig)
from nnabla_rl.model_trainers.q_value.clipped_double_q_trainer import (  # noqa
    ClippedDoubleQTrainer, ClippedDoubleQTrainerConfig)
from nnabla_rl.model_trainers.q_value.ddpg_q_trainer import (  # noqa
    DDPGQTrainer, DDPGQTrainerConfig)
from nnabla_rl.model_trainers.q_value.ddqn_q_trainer import (  # noqa
    DDQNQTrainer, DDQNQTrainerConfig)
from nnabla_rl.model_trainers.q_value.dqn_q_trainer import (  # noqa
    DQNQTrainer, DQNQTrainerConfig)
from nnabla_rl.model_trainers.q_value.iqn_q_trainer import (  # noqa
    IQNQTrainer, IQNQTrainerConfig)
from nnabla_rl.model_trainers.q_value.munchausen_rl_q_trainer import (  # noqa
    MunchausenIQNQTrainer, MunchausenIQNQTrainerConfig, MunchausenDQNQTrainer, MunchausenDQNQTrainerConfig)
from nnabla_rl.model_trainers.q_value.qrdqn_q_trainer import (  # noqa
    QRDQNQTrainer, QRDQNQTrainerConfig)
from nnabla_rl.model_trainers.q_value.soft_q_trainer import (  # noqa
    SoftQTrainer, SoftQTrainerConfig)
from nnabla_rl.model_trainers.q_value.td3_q_trainer import (  # noqa
    TD3QTrainer, TD3QTrainerConfig)
from nnabla_rl.model_trainers.q_value.v_targeted_q_trainer import (  # noqa
    VTargetedQTrainer, VTargetedQTrainerConfig)
