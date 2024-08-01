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

import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.utils import serializers


def main():
    config = A.DDPGConfig(start_timesteps=200)
    env = E.DummyContinuous()
    ddpg = A.DDPG(env, config=config)

    outdir = "./save_load_snapshot"

    # This actually saves the model and solver state right after the algorithm construction
    snapshot_dir = serializers.save_snapshot(outdir, ddpg)

    # This actually loads the model and solver state which is saved with the code above
    algorithm = serializers.load_snapshot(snapshot_dir, env)
    assert isinstance(algorithm, A.DDPG)


if __name__ == "__main__":
    main()
