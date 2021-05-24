# Contributing to NNablaRL

Contributions of any kind are welcome! Check below contribution options and select the option that fits you best.

## Contributing by thumbing up to existing issues and pull requests

We encourage thumbing up :+1: to good issues and/or pull requests in high demand.

## Contributing by posting issues/proposals

If you encounter any bugs, or come up with any feature/algorithm requests, check the [issue-tracker](https://github.com/sony/nnabla-rl/issues). If you can not find any existing issue, please feel free to post a new issue.  
Please do **NOT** post questions about the library, such as usage, installation, etc., to the issue-tracker. Use [NNabla user group](https://groups.google.com/forum/#!forum/nnabla) for such questions.

## Contributing by improving the document

If you find any typo, gramatical error, incorrect explanation etc. in [NNablaRL's documentation](https://github.com/sony/nnabla-rl/docs) or READMEs follow the below procedure and send pull request!

1. Search existing issues and/or pull requests in the [NNablaRL GitHub repository](https://github.com/sony/nnabla-rl).
2. If doesn't exist, post an issue for the improvement proposal.
3. Fork the repository, and improve the document.
4. (If you improve the NNablaRL's documentation) Check that the document successfully builds and properly displayed. (See: [How to build the document](#how-to-build-the-document) section to build the document on your machine)
5. Create a pull request of your development branch to NNablaRL's master branch. Our maintainers will then review your changes.
6. Once your change is accepted, our maintainer will merge your change.

### How to build the document

To build the documentation, you will need [Sphinx](http://www.sphinx-doc.org) and some additional python packages.

```sh
cd docs/
pip install -r requirements.txt
```

You can then build the documentation by running ``make <format>`` from the
``docs/`` folder. Run ``make`` to get a list of all available output formats.

We recommend building the document as html.

```sh
cd docs/
make html
```

## Contributing code

We appreciate contributors in the community, that are willing to improve NNablaRL. We follow the development style used in [NNabla](https://github.com/sony/nnabla) listed below.

1. Search existing issues and/or pull requests in the [NNablaRL GitHub repository](https://github.com/sony/nnabla-rl).
2. If doesn't exist, post an issue for the feature proposal.
3. Fork the repository, and develop your feature.
4. Format your code according to the NNablaRL's coding style. (See: [Code format guidelines](#code-format-guidelines) section below for details)
5. Write an unit test(s) and also check that linters do not raise any error. If you implement a deep reinforcement learning algorithm, please also check that your implementation reproduces the result presented in the paper that you referred. (See: [Testing guidelines](#testing-guidelines) section below for details)
6. Create a pull request of your development branch to NNablaRL's master branch. Our maintainers will then review your changes.
7. Once your change is accepted, our maintainer will merge your change.

**NOTE**: Before starting to develop NNablaRL's code, install extra python packages that will be used for code formatting and testing. You can install extra packages as follows.

```sh
cd <nnabla-rl root directory>
pip install -r requirements.txt
```

We also recommend installing the NNablaRL package as follows to reflect code changes made during the development automatically.

```sh
$ cd <nnabla-rl root directory>
$ pip install -e .
```

### Code format guidelines

We use [autopep8](https://github.com/hhatto/autopep8) and [isort](https://github.com/PyCQA/isort) to keep consistent coding style. After finishing developing the code, run autopep8 and isort to ensure that your code is correctly formatted.

You can run autopep8 and isort as follows.

```sh
cd <nnabla-rl root directory>
autopep8 .
```

```sh
cd <nnabla-rl root directory>
isort .
```

### Testing guidelines

#### Writing unit test

If there is no existing test that checks your changes, please write a test(s) to check the validity of your code. Any pull request without unit test will **NOT** be accepted.  
When adding a new unit test file, place the unit test file under the tests/ directory with name test_\<the file name to test\>.py. See the below example.

Example: When adding tests for your_new_file.py placed under nnabla_rl/utils.

```sh
.
├── ./nnabla_rl
│   └── ./nnabla_rl/utils
│       └── ./nnabla_rl/utils/your_new_file.py
└── ./tests
    └── ./tests/utils
        └── ./tests/utils/test_your_new_file.py
```

You can run tests with the following command.

```sh
cd <nnabla-rl root directory>
python setup.py test
```

#### Evaluating the algorithm

In case your pull request contains a new implementation of deep reinforcement learning algorithm, please check that your implementation reproduces the original paper's result and include the result that you obtained in the pull request comment. Please also provide a python script that reproduces the result and a README.md file that summarizes the evaluation result. Place the script used for reproduction and README.md under reproductions/ directory as follows.


```sh
.
└── ./reproductions
    └── ./reproductions/<evaluated_env>
        └── ./reproductions/<evaluated_env>/<algorithm_name>
            └── ./reproductions/<evaluated_env>/<algorithm_name>/<algorithm_name>_reproduction.py
            └── ./reproductions/<evaluated_env>/<algorithm_name>/README.md
```

If you can not find an appropriate benchmark, scores, dataset, etc. for the implemented algorithm (for example, the dataset used in the paper is inaccessible), please consider evaluating the implementation using an alternative environment and provide the evaluation result obtained in the alternative environment.

#### Linting

We use [flake8](https://gitlab.com/pycqa/flake8) and [mypy](https://github.com/python/mypy) to check code consistency and type annotations. Run flake8 and mypy to check that your implementation does not raise any error.

```sh
cd <nnabla-rl root directory>
flake8
```

```sh
cd <nnabla-rl root directory>
mypy
```
