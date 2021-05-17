# Copyright 2020,2021 Sony Corporation.
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

from nnabla_rl.algorithm import Algorithm, AlgorithmConfig
from nnabla_rl.algorithms.a2c import A2C, A2CConfig
from nnabla_rl.algorithms.bcq import BCQ, BCQConfig
from nnabla_rl.algorithms.bear import BEAR, BEARConfig
from nnabla_rl.algorithms.categorical_dqn import CategoricalDQN, CategoricalDQNConfig
from nnabla_rl.algorithms.ddpg import DDPG, DDPGConfig
from nnabla_rl.algorithms.dqn import DQN, DQNConfig
from nnabla_rl.algorithms.dummy import Dummy, DummyConfig
from nnabla_rl.algorithms.gail import GAIL, GAILConfig
from nnabla_rl.algorithms.icml2015_trpo import ICML2015TRPO, ICML2015TRPOConfig
from nnabla_rl.algorithms.icml2018_sac import ICML2018SAC, ICML2018SACConfig
from nnabla_rl.algorithms.iqn import IQN, IQNConfig
from nnabla_rl.algorithms.munchausen_dqn import MunchausenDQN, MunchausenDQNConfig
from nnabla_rl.algorithms.munchausen_iqn import MunchausenIQN, MunchausenIQNConfig
from nnabla_rl.algorithms.ppo import PPO, PPOConfig
from nnabla_rl.algorithms.qrdqn import QRDQN, QRDQNConfig
from nnabla_rl.algorithms.reinforce import REINFORCE, REINFORCEConfig
from nnabla_rl.algorithms.sac import SAC, SACConfig
from nnabla_rl.algorithms.td3 import TD3, TD3Config
from nnabla_rl.algorithms.trpo import TRPO, TRPOConfig

# Do NOT manipulate this dictionary directly.
# Use register_algorithm() instead.
_ALGORITHMS = {}


def register_algorithm(algorithm_class, config_class):
    global _ALGORITHMS
    if not issubclass(algorithm_class, Algorithm):
        raise ValueError(
            "{} is not subclass of Algorithm".format(algorithm_class))
    if not issubclass(config_class, AlgorithmConfig):
        raise ValueError(
            "{} is not subclass of AlgorithmConfig".format(config_class))
    _ALGORITHMS[algorithm_class.__name__] = (algorithm_class, config_class)


def is_registered(algorithm_class, config_class):
    for registered in _ALGORITHMS.values():
        if algorithm_class in registered and config_class in registered:
            return True
    return False


def get_class_of(name):
    return _ALGORITHMS[name]


register_algorithm(A2C, A2CConfig)
register_algorithm(BCQ, BCQConfig)
register_algorithm(BEAR, BEARConfig)
register_algorithm(CategoricalDQN, CategoricalDQNConfig)
register_algorithm(DDPG, DDPGConfig)
register_algorithm(DQN, DQNConfig)
register_algorithm(Dummy, DummyConfig)
register_algorithm(ICML2018SAC, ICML2018SACConfig)
register_algorithm(IQN, IQNConfig)
register_algorithm(MunchausenDQN, MunchausenDQNConfig)
register_algorithm(MunchausenIQN, MunchausenIQNConfig)
register_algorithm(PPO, PPOConfig)
register_algorithm(QRDQN, QRDQNConfig)
register_algorithm(REINFORCE, REINFORCEConfig)
register_algorithm(SAC, SACConfig)
register_algorithm(TD3, TD3Config)
register_algorithm(ICML2015TRPO, ICML2015TRPOConfig)
register_algorithm(TRPO, TRPOConfig)
register_algorithm(GAIL, GAILConfig)
