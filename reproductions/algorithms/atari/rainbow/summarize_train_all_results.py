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
import argparse
import pathlib
import re
import subprocess


def find_tsvs(result_dir: pathlib.Path):
    print(f'result dir {result_dir}')
    tsvs = []
    for f in result_dir.iterdir():
        if f.is_dir():
            tsvs.extend(find_tsvs(f))
        elif f.name == 'evaluation_result_average_scalar.tsv':
            tsvs.append(f)
    tsvs.sort()
    return tsvs


def extract_label(tsv_path: pathlib.Path):
    pattern = r'.*-v4[- ](.*)_results'
    regex = re.compile(pattern)
    env_method = str(tsv_path).split('/')[2]
    print(env_method)
    result = regex.findall(env_method)
    if len(result) == 0:
        return 'rainbow'
    else:
        return result[0]


def extract_labels(tsv_paths):
    labels = []
    for tsv in tsv_paths:
        label = extract_label(tsv)
        labels.append(label)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    result_dir = pathlib.Path(args.result_dir)
    tsvs = find_tsvs(result_dir)
    labels = extract_labels(tsvs)

    tsvroot = tsvs[0].parent.parent
    tsvpaths = [str(tsv) for tsv in tsvs]
    tsvlabels = labels
    print(f'tsvroot: {tsvroot}')
    print(f'tsvs: {tsvs}')
    print(f'labels: {labels}')

    command = ['plot_result', '--tsvpaths'] + tsvpaths + \
        ['--tsvlabels'] + tsvlabels + \
        ['--no-stddev'] + \
        ['--smooth-k'] + ['10'] + \
        ['--outdir'] + [f'{str(tsvroot)}']
    print(f'command: {command}')
    subprocess.run(command)


if __name__ == "__main__":
    main()
