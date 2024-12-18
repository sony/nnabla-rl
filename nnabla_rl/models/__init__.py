# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

from nnabla_rl.models.decision_transformer import (  # noqa
    DecisionTransformer,
    DeterministicDecisionTransformer,
    StochasticDecisionTransformer,
)
from nnabla_rl.models.distributional_function import (  # noqa
    ValueDistributionFunction,
    DiscreteValueDistributionFunction,
    ContinuousValueDistributionFunction,
)
from nnabla_rl.models.distributional_function import (  # noqa
    QuantileDistributionFunction,
    DiscreteQuantileDistributionFunction,
    ContinuousQuantileDistributionFunction,
)
from nnabla_rl.models.distributional_function import (  # noqa
    StateActionQuantileFunction,
    DiscreteStateActionQuantileFunction,
    ContinuousStateActionQuantileFunction,
)
from nnabla_rl.models.dynamics import Dynamics, DeterministicDynamics, StochasticDynamics  # noqa
from nnabla_rl.models.model import Model  # noqa
from nnabla_rl.models.perturbator import Perturbator  # noqa
from nnabla_rl.models.policy import Policy, DeterministicPolicy, StochasticPolicy  # noqa
from nnabla_rl.models.q_function import (  # noqa
    QFunction,
    DiscreteQFunction,
    ContinuousQFunction,
    FactoredContinuousQFunction,
)
from nnabla_rl.models.v_function import VFunction  # noqa
from nnabla_rl.models.reward_function import RewardFunction  # noqa
from nnabla_rl.models.encoder import Encoder, VariationalAutoEncoder  # noqa
from nnabla_rl.models.intra_policy import IntraPolicy, StochasticIntraPolicy  # noqa
from nnabla_rl.models.termination_function import TerminationFunction, StochasticTerminationFunction  # noqa
from nnabla_rl.models.option_value_function import OptionValueFunction, DiscreteOptionValueFunction  # noqa

from nnabla_rl.models.mujoco.policies import TD3Policy, SACPolicy, BEARPolicy, TRPOPolicy  # noqa
from nnabla_rl.models.mujoco.q_functions import (  # noqa
    TD3QFunction,
    SACQFunction,
    SACDQFunction,
    HERQFunction,
    XQLQFunction,
    IQLQFunction,
)
from nnabla_rl.models.mujoco.decision_transformers import MujocoDecisionTransformer  # noqa
from nnabla_rl.models.mujoco.distributional_functions import QRSACQuantileDistributionFunction  # noqa
from nnabla_rl.models.mujoco.v_functions import SACVFunction, TRPOVFunction, ATRPOVFunction  # noqa
from nnabla_rl.models.mujoco.v_functions import PPOVFunction as PPOMujocoVFunction  # noqa
from nnabla_rl.models.mujoco.v_functions import GAILVFunction  # noqa
from nnabla_rl.models.mujoco.v_functions import XQLVFunction  # noqa
from nnabla_rl.models.mujoco.v_functions import IQLVFunction  # noqa
from nnabla_rl.models.mujoco.encoders import UnsquashedVariationalAutoEncoder, BCQVariationalAutoEncoder  # noqa
from nnabla_rl.models.mujoco.perturbators import BCQPerturbator  # noqa
from nnabla_rl.models.mujoco.policies import ICML2015TRPOPolicy as ICML2015TRPOMujocoPolicy  # noqa
from nnabla_rl.models.mujoco.policies import PPOPolicy as PPOMujocoPolicy  # noqa
from nnabla_rl.models.mujoco.policies import GAILPolicy  # noqa
from nnabla_rl.models.mujoco.policies import HERPolicy  # noqa
from nnabla_rl.models.mujoco.policies import ATRPOPolicy  # noqa
from nnabla_rl.models.mujoco.policies import XQLPolicy  # noqa
from nnabla_rl.models.mujoco.policies import IQLPolicy  # noqa
from nnabla_rl.models.mujoco.reward_functions import GAILDiscriminator  # noqa
from nnabla_rl.models.atari.decision_transformers import AtariDecisionTransformer  # noqa
from nnabla_rl.models.atari.policies import PPOPolicy as PPOAtariPolicy  # noqa
from nnabla_rl.models.atari.policies import A3CPolicy  # noqa
from nnabla_rl.models.atari.q_functions import DQNQFunction, DRQNQFunction  # noqa
from nnabla_rl.models.atari.v_functions import PPOVFunction as PPOAtariVFunction  # noqa
from nnabla_rl.models.atari.v_functions import A3CVFunction  # noqa
from nnabla_rl.models.atari.shared_functions import (  # noqa
    PPOSharedFunctionHead,
    A3CSharedFunctionHead,
    OptionCriticSharedFunctionHead,
)
from nnabla_rl.models.atari.distributional_functions import (  # noqa
    C51ValueDistributionFunction,
    RainbowValueDistributionFunction,
    RainbowNoDuelValueDistributionFunction,
    RainbowNoNoisyValueDistributionFunction,
    QRDQNQuantileDistributionFunction,
    IQNQuantileFunction,
)
from nnabla_rl.models.atari.policies import ICML2015TRPOPolicy as ICML2015TRPOAtariPolicy  # noqa
from nnabla_rl.models.atari.intra_policies import AtariOptionCriticIntraPolicy  # noqa
from nnabla_rl.models.atari.termination_functions import AtariOptionCriticTerminationFunction  # noqa
from nnabla_rl.models.atari.option_v_functions import AtariOptionCriticOptionVFunction  # noqa

from nnabla_rl.models.pybullet.q_functions import ICRA2018QtOptQFunction  # noqa
from nnabla_rl.models.pybullet.reward_functions import AMPDiscriminator  # noqa
from nnabla_rl.models.pybullet.policy import AMPGatedPolicy, AMPPolicy  # noqa
from nnabla_rl.models.pybullet.v_functions import AMPGatedVFunction, AMPVFunction  # noqa

from nnabla_rl.models.classic_control.policies import REINFORCEContinousPolicy, REINFORCEDiscretePolicy  # noqa
from nnabla_rl.models.classic_control.dynamics import MPPIDeterministicDynamics  # noqa

from nnabla_rl.models.hybrid_env.encoders import HyARVAE  # noqa
from nnabla_rl.models.hybrid_env.policies import HyARPolicy  # noqa
from nnabla_rl.models.hybrid_env.q_functions import HyARQFunction  # noqa
