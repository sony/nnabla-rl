# Copyright 2022 Sony Group Corporation.
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

import os
import tempfile

import pytest

import nnabla as nn
from nnabla.utils.nnp_graph import NnpLoader
from nnabla_rl.algorithms.ddpg import DDPG, DDPGConfig
from nnabla_rl.environments.dummy import DummyContinuous
from nnabla_rl.hooks import TrainingGraphHook


class TestComputationalGraphHook():
    def test_call(self):
        nn.clear_parameters()
        with tempfile.TemporaryDirectory() as dname:
            env = DummyContinuous()

            hook = TrainingGraphHook(outdir=dname, name="training")

            # save nntxt
            ddpg = DDPG(env_or_env_info=env)
            ddpg.set_hooks([hook])
            ddpg.train_online(env, 1)

            # load nntxt
            nnp = NnpLoader(os.path.join(dname, "training.nntxt"))

            # check networks
            trainer_names = list(ddpg.trainers.keys())
            assert nnp.get_network_names() == trainer_names

    def test_rnn_support(self):
        nn.clear_parameters()
        with tempfile.TemporaryDirectory() as dname:
            env = DummyContinuous()

            hook = TrainingGraphHook(outdir=dname, name="training")

            # save nntxt
            config = DDPGConfig(actor_unroll_steps=2, critic_unroll_steps=2)
            ddpg = DDPG(env_or_env_info=env, config=config)
            ddpg.set_hooks([hook])
            ddpg.train_online(env, 1)

            # load nntxt
            nnp = NnpLoader(os.path.join(dname, "training.nntxt"))

            # check networks
            trainer_names = list(ddpg.trainers.keys())
            assert nnp.get_network_names() == trainer_names


if __name__ == "__main__":
    pytest.main()
