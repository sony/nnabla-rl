from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.algorithms.bcq import BCQ, BCQParam
from nnabla_rl.algorithms.bear import BEAR, BEARParam
from nnabla_rl.algorithms.categorical_dqn import CategoricalDQN, CategoricalDQNParam
from nnabla_rl.algorithms.ddpg import DDPG, DDPGParam
from nnabla_rl.algorithms.dqn import DQN, DQNParam
from nnabla_rl.algorithms.dummy import Dummy, DummyParam
from nnabla_rl.algorithms.icml2018_sac import ICML2018SAC, ICML2018SACParam
from nnabla_rl.algorithms.iqn import IQN, IQNParam
from nnabla_rl.algorithms.munchausen_dqn import MunchausenDQN, MunchausenDQNParam
from nnabla_rl.algorithms.munchausen_iqn import MunchausenIQN, MunchausenIQNParam
from nnabla_rl.algorithms.ppo import PPO, PPOParam
from nnabla_rl.algorithms.qrdqn import QRDQN, QRDQNParam
from nnabla_rl.algorithms.reinforce import REINFORCE, REINFORCEParam
from nnabla_rl.algorithms.sac import SAC, SACParam
from nnabla_rl.algorithms.td3 import TD3, TD3Param
from nnabla_rl.algorithms.icml2015_trpo import ICML2015TRPO, ICML2015TRPOParam
from nnabla_rl.algorithms.trpo import TRPO, TRPOParam

# Do NOT manipulate this dictionary directly.
# Use register_algorithm() instead.
_ALGORITHMS = {}


def register_algorithm(algorithm_class, param_class):
    global _ALGORITHMS
    if not issubclass(algorithm_class, Algorithm):
        raise ValueError(
            "{} is not subclass of Algorithm".format(algorithm_class))
    if not issubclass(param_class, AlgorithmParam):
        raise ValueError(
            "{} is not subclass of AlgorithmParam".format(param_class))
    _ALGORITHMS[algorithm_class.__name__] = (algorithm_class, param_class)


def is_registered(algorithm_class, param_class):
    for registered in _ALGORITHMS.values():
        if algorithm_class in registered and param_class in registered:
            return True
    return False


def get_class_of(name):
    return _ALGORITHMS[name]


register_algorithm(BCQ, BCQParam)
register_algorithm(BEAR, BEARParam)
register_algorithm(CategoricalDQN, CategoricalDQNParam)
register_algorithm(DDPG, DDPGParam)
register_algorithm(DQN, DQNParam)
register_algorithm(Dummy, DummyParam)
register_algorithm(ICML2018SAC, ICML2018SACParam)
register_algorithm(IQN, IQNParam)
register_algorithm(MunchausenDQN, MunchausenDQNParam)
register_algorithm(MunchausenIQN, MunchausenIQNParam)
register_algorithm(PPO, PPOParam)
register_algorithm(QRDQN, QRDQNParam)
register_algorithm(REINFORCE, REINFORCEParam)
register_algorithm(SAC, SACParam)
register_algorithm(TD3, TD3Param)
register_algorithm(ICML2015TRPO, ICML2015TRPOParam)
register_algorithm(TRPO, TRPOParam)
