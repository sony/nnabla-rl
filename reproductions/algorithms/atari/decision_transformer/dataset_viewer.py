# Copyright 2023 Sony Group Corporation.
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

import argparse
import pathlib

import cv2
from atari_dataset_loader import load_expert_dataset


def view_dataset(args):
    dataset_dir = pathlib.Path(args.dataset_dir)
    (o, a, r, t) = load_expert_dataset(dataset_dir)
    print('observation data shape: ', o.shape)
    print('action data shape: ', a.shape)
    print('reward data shape: ', r.shape)
    print('terminal data shape: ', t.shape)

    show_observations(o)


def show_observations(observations):
    print('press q to quit displaying observation')
    for observation in observations:
        cv2.imshow('obs0', observation)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='datasets/Breakout/1/replay_logs')
    args = parser.parse_args()

    view_dataset(args)


if __name__ == '__main__':
    main()
