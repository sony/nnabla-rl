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

import argparse
import pathlib
import shutil
from csv import reader

import numpy as np

import nnabla_rl.writers as W


def load_histogram_data(path, dtype=float):
    histogram = []
    with open(path) as f:
        tsv_reader = reader(f, delimiter="\t")
        for i, row in enumerate(tsv_reader):
            if i == 0:
                continue
            histogram.append([row[0], np.asarray(row[1:])])
    return histogram


def rmdir_if_exists(path):
    if path.exists():
        shutil.rmtree(path)


def list_all_directory_with(rootdir, filename):
    directories = []
    for f in rootdir.iterdir():
        if f.is_file():
            if f.name in filename:
                directories.append(f.parent)
        if f.is_dir():
            directories.extend(list_all_directory_with(f, filename))
    return directories


def extract_iteration_num_and_returns(histogram_data):
    iteration_nums = []
    returns = []
    for i in range(len(histogram_data)):
        data_row = histogram_data[i]
        if "returns" in data_row[0]:
            iteration_nums.append(int(data_row[0].split(" ")[0]))
            scores = data_row[1][0:].astype(float)
            returns.append(scores)

    return iteration_nums, returns


def create_gamer_normalized_score_file(histograms, file_outdir, gamer_score):
    iteration_nums = None
    returns_list = None
    for histogram in histograms:
        data = load_histogram_data(histogram, dtype=str)
        iteration_nums, returns = extract_iteration_num_and_returns(data)
        if returns_list is None:
            returns_list = returns
        else:
            for i, r in enumerate(returns):
                returns_list[i] = np.concatenate([returns_list[i], r])

    file_prefix = "gamer_normalized_score.tsv"
    writer = W.FileWriter(file_outdir, file_prefix=file_prefix)
    for i, r in zip(iteration_nums, returns_list):
        normalized_r = r / gamer_score
        mean = np.mean(normalized_r) * 100
        std_dev = np.std(normalized_r) * 100

        scalar_results = {}
        scalar_results["mean"] = mean
        scalar_results["std_dev"] = std_dev

        writer.write_scalar(i, scalar_results)


def compile_results(args):
    rootdir = pathlib.Path.cwd()

    histograms = {}
    histogram_directories = list_all_directory_with(rootdir, args.eval_histogram_filename)
    print(f"files: {histogram_directories}")
    for directory in histogram_directories:
        if args.resultdir not in str(directory):
            continue
        relative_dir = directory.relative_to(rootdir)
        env_name = str(relative_dir).split("/")[0]
        histogram_file = directory / args.eval_histogram_filename
        print(f"found histogram file of env: {env_name} at: {histogram_file}")
        if histogram_file.exists():
            if env_name in histograms.keys():
                histograms[env_name].append(histogram_file)
            else:
                histograms[env_name] = [histogram_file]

    for env_name, histograms in histograms.items():
        file_outdir = pathlib.Path(args.outdir) / pathlib.Path(env_name)
        create_gamer_normalized_score_file(histograms, file_outdir, args.gamer_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, required=True, help="output directory")
    parser.add_argument("--resultdir", type=str, required=True, help="result directory")
    parser.add_argument("--gamer-score", type=float, required=True, help="gamer score")
    parser.add_argument(
        "--eval-histogram-filename",
        type=str,
        default="evaluation_result_histogram.tsv",
        help="eval result(histogram) filename",
    )

    args = parser.parse_args()

    compile_results(args)
