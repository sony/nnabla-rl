# Copyright 2024 Sony Group Corporation.
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

import numpy as np
from rliable import library as rly
from rliable import metrics

DMC15_TASKS = [
    'acrobot-swingup',
    'cheetah-run',
    'finger-turn_hard',
    'fish-swim',
    'hopper-hop',
    'hopper-stand',
    'humanoid-run',
    'humanoid-stand',
    'humanoid-walk',
    'pendulum-swingup',
    'quadruped-run',
    'quadruped-walk',
    'reacher-hard',
    'swimmer-swimmer6',
    'walker-run'
]


def find_data_files(path):
    item = pathlib.Path(path)
    data_files = []
    if item.is_file():
        data_files.append(item)
    else:
        for d in sorted(item.iterdir()):
            data_files.extend(find_data_files(d))
    return data_files


def extract_task_name(path: pathlib.Path):
    for task in DMC15_TASKS:
        if task in str(path):
            return task
    return None


def read_score(f: pathlib.Path):
    if f.suffix == ".txt":
        return np.loadtxt(f)[:100, 1][-1]
    if f.suffix == ".tsv":
        if f.name != "evaluation_result_scalar.tsv":
            return None
        if "seed" not in str(f):
            return None
        return np.loadtxt(f, delimiter="\t", skiprows=1)[:, 1][-1]
    return None


def load_training_data(path):
    files = find_data_files(path)

    data_dict = {task: [] for task in DMC15_TASKS}
    for f in files:
        task_name = extract_task_name(f)
        if task_name is None:
            continue
        data = read_score(f)
        if data is None:
            continue
        data_dict[task_name].append(np.round(data, 1))
    return data_dict


def data_to_matrix(data):
    return np.stack([v for v in data.values()], axis=1)


def evaluate(args):
    training_data = load_training_data(args.rootdir)
    print(f'training_data: {training_data}')
    data_matrix = data_to_matrix(training_data)
    print(f'data_matrix: {data_matrix}. shape: {data_matrix.shape}')

    # data should be num_runs x num_tasks
    data = {'score': data_matrix}

    def aggregate_func(x):
        return np.array([metrics.aggregate_iqm(x),
                         metrics.aggregate_median(x),
                         metrics.aggregate_mean(x)])
    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        data, aggregate_func, reps=50000
    )
    print(f'aggregate scores: {aggregate_scores}')
    print(f'aggregate cis: {aggregate_cis}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, required=True)
    parser.add_argument('--target-file-name', type=str, default=None)

    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
