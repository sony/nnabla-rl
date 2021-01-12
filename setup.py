from setuptools import find_packages
from setuptools import setup

install_requires = ['nnabla>=1.14', 'gym', 'dataclasses;python_version=="3.6"', 'opencv-python']

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
