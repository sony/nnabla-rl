[![Build status](https://github.com/nnabla/nnabla-rl/workflows/Build%20nnabla-rl/badge.svg)](https://github.com/nnabla/nnabla-rl/actions)

# Deep Reinforcement Learning Library built on top of Neural Network Libraries

NNablaRL is a deep reinforcement learning library built on top of [Neural Network Libraries](https://github.com/sony/nnabla) 
that is intended to be used for research, development and production.

## Installation

Installing NNablaRL is easy!

```sh
$ pip install nnabla_rl
```

If you would like to install nnabla_rl for development

```sh
$ cd <nnabla_rl root dir>
$ pip install -e .
```

NNablaRL only supports Python version >= 3.6 and [NNabla](https://github.com/sony/nnabla) version >= 1.17.

## Features

### Friendly API

NNablaRL has friendly Python APIs which enables to start training with only 3 lines of python code.

```py
import nnabla_rl
import nnabla_rl.algorithms as A
from nnabla_rl.utils.reproductions import build_atari_env

env = build_atari_env("BreakoutNoFrameskip-v4") # 1
dqn = A.DQN(env)  # 2
dqn.train(env)  # 3
```

To get more details about NNablaRL, see documentation and [examples](./examples).

### Builtin Algorithms

See [algorithms](./nnabla_rl/algorithms/README.md).

You can find the reproduction and evaluation results of each algorithm [here](./reproductions)  
Note that you may not get completely the same results when running the reproduction code on your computer. The result may slightly change depending on your machine, nnabla/nnabla-rl's package version, etc.

## Getting started

Try below interactive demos to get started. </br>
You can run it directly on [Colab](https://colab.research.google.com/) from the links in the table below.

| Title | Notebook | Target RL task |
|:---:|:---:|:---:|
| Simple reinforcement learning training to get started | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-rl/blob/master/interactive-demos/pendulum.ipynb) | Pendulum |
| Learn how to use training algorithms | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-rl/blob/master/interactive-demos/tutorial-algorithm.ipynb) | Pendulum |
| Learn how to use customized network model for training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-rl/blob/master/interactive-demos/tutorial-model.ipynb) | Mountain car |
| Learn how to use different network solver for training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-rl/blob/master/interactive-demos/tutorial-solver.ipynb) | Pendulum |
| Learn how to use different replay buffer for training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-rl/blob/master/interactive-demos/tutorial-replay-buffer.ipynb) | Pendulum |
| Learn how to use your own environment for training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-rl/blob/master/interactive-demos/tutorial-envs.ipynb) | Customized environment |
| Atari game training example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-rl/blob/master/interactive-demos/atari.ipynb) | Atari games |

## Documentation

Full documentation is under the docs/.
To build the documentation, you will need [Sphinx](http://www.sphinx-doc.org) and some additional python packages.

```
cd docs/
pip install -r requirements.txt
```

You can then build the documentation by running ``make <format>`` from the
``docs/`` folder. Run ``make`` to get a list of all available output formats.

## Contribution guide

TBD.

## License

NNablaRL is provided under the [Apache License Version 2.0](LICENSE) license.
