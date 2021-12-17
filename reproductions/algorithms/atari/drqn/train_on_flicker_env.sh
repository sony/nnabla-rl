#!/usr/bin/env bash
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

if [ $# -ne 2 ]; then
    echo "usage: $0 <gpu_id> <flicker>"
    exit 1
fi

# env list
ENVS=(
    "AsteroidsNoFrameskip-v4"
    "BeamRiderNoFrameskip-v4"
    "BowlingNoFrameskip-v4"
    "CentipedeNoFrameskip-v4"
    "ChopperCommandNoFrameskip-v4"
    "DoubleDunkNoFrameskip-v4"
    "FrostbiteNoFrameskip-v4"
    "IceHockeyNoFrameskip-v4"
    "MsPacmanNoFrameskip-v4"
    "PongNoFrameskip-v4"
)

for ENV in ${ENVS[@]}
do
    ./train_drqn.sh $1 $ENV $2 &
done
