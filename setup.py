# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

from setuptools import find_packages, setup

install_requires = ['nnabla>=1.17', 'gym', 'dataclasses;python_version=="3.6"', 'opencv-python']

tests_require = ['pytest', 'pytest-cov', 'mock']
# pytest-runner is required to run tests with
# $ python setup.py tests
setup_requires = ['pytest-runner']

scripts = ['bin/plot_result',
           'bin/check_best_iteration',
           'bin/compile_results',
           'bin/train_and_compile_results']
description = '''Deep reinforcement learning framework that is intended \
                 to be used for research, development and production.'''
setup(
    name='nnabla_rl',
    version='0.0.1',
    description=description,
    author='Yu Ishihara',
    author_email='yu.ishihara@sony.com',
    install_requires=install_requires,
    packages=find_packages(exclude=('examples', 'reproductions', 'tests')),
    scripts=scripts,
    setup_requires=setup_requires,
    python_requires='>=3.6',
    test_suite='tests',
    tests_require=tests_require
)
