# Copyright 2023,2024 Sony Group Corporation.
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

import gzip
import pathlib
from concurrent import futures

import numpy as np


def load_data_from_gz(gzfile):
    with gzip.open(gzfile, mode="rb") as f:
        data = np.load(f, allow_pickle=False)
    return data


def find_all_file_with_name(path, name):
    path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
    file_paths = []
    for f in path.iterdir():
        if f.is_dir():
            paths = find_all_file_with_name(f, name)
            file_paths.extend(paths)
        if f.is_file() and name in f.name:
            file_paths.append(f)
    return file_paths


def load_experience(filepath):
    return load_data_from_gz(filepath)


def load_experiences(gzfiles, num_experiences_to_load):
    experiences = None
    for f in gzfiles:
        experience = load_experience(f)
        if experiences is None:
            experiences = experience
        else:
            experiences.extend(experience)
        print("loaded experiences: {} / {}".format(len(experiences), num_experiences_to_load))
        if num_experiences_to_load <= len(experiences):
            break
    return experiences[:num_experiences_to_load]


def load_dataset(dataset_dir, percentage):
    dataset_dir = pathlib.Path(dataset_dir)
    observation_files = find_all_file_with_name(dataset_dir, "observation")
    observation_files.sort()

    action_files = find_all_file_with_name(dataset_dir, "action")
    action_files.sort()

    reward_files = find_all_file_with_name(dataset_dir, "reward")
    reward_files.sort()

    terminal_files = find_all_file_with_name(dataset_dir, "terminal")
    terminal_files.sort()

    file_num = len(observation_files)
    total_experiences = 1e6 * file_num
    total_num_to_load = int(total_experiences * percentage) // 100

    observations = load_experiences(observation_files, total_num_to_load)
    actions = load_experiences(action_files, total_num_to_load)
    rewards = load_experiences(reward_files, total_num_to_load)
    terminals = load_experiences(terminal_files, total_num_to_load)

    return observations, actions, rewards, terminals


def load_dataset_by_dataset_num(dataset_dir, dataset_num):
    dataset_dir = pathlib.Path(dataset_dir)
    observation_file = dataset_dir / f"$store$_observation_ckpt.{dataset_num}.gz"
    action_file = dataset_dir / f"$store$_action_ckpt.{dataset_num}.gz"
    reward_file = dataset_dir / f"$store$_reward_ckpt.{dataset_num}.gz"
    terminal_file = dataset_dir / f"$store$_terminal_ckpt.{dataset_num}.gz"

    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        data_futures = [
            executor.submit(load_experience, file_name)
            for file_name in (observation_file, action_file, reward_file, terminal_file)
        ]
        observations = data_futures[0].result()
        actions = data_futures[1].result()
        rewards = data_futures[2].result()
        terminals = data_futures[3].result()

    return observations, actions, rewards, terminals


def load_expert_dataset(dataset_dir):
    return load_dataset_by_dataset_num(dataset_dir, 50)
