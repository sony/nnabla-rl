[![License](https://img.shields.io/github/license/sony/nnabla-rl)](LICENSE)
[![Build status](https://github.com/nnabla/nnabla-rl/workflows/Build%20nnabla-rl/badge.svg)](https://github.com/nnabla/nnabla-rl/actions)

# Deep Reinforcement Learning Library built on top of Neural Network Libraries

NNablaRL is a deep reinforcement learning library built on top of [Neural Network Libraries](https://github.com/sony/nnabla) 
that is intended to be used for research, development and production.

## Installation

Installing NNablaRL is easy!

```sh
$ pip install nnabla-rl
```

NNablaRL only supports Python version >= 3.6 and [NNabla](https://github.com/sony/nnabla) version >= 1.17.

### Enabling GPU accelaration (Optional)

NNablaRL algorithms run on CPU by default. To run the algorithm on GPU, first install [nnabla-ext-cuda](https://github.com/sony/nnabla-ext-cuda) as follows.
(Replace [cuda-version] depending on the CUDA version installed on your machine.)

```sh
$ pip install nnabla-ext-cuda[cuda-version]
```

```sh
# Example installation. Supposing CUDA 11.0 is installed on your machine.
$ pip install nnabla-ext-cuda110
```

After installing nnabla-ext-cuda, set the gpu id to run the algorithm on through algorithm's configuration.

```py
import nnabla_rl.algorithms as A

config = A.DQNConfig(gpu_id=0) # Use gpu 0. If negative, will run on CPU.
dqn = A.DQN(env, config=config)
...
```

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

To get more details about NNablaRL, see [documentation](https://nnabla-rl.readthedocs.io/) and [examples](./examples).

### Many builtin algorithms

Most of famous/SOTA deep reinforcement learning algorithms, such as DQN, SAC, BCQ, GAIL, etc., are implemented in NNablaRL. Implemented algorithms are carefully tested and evaluated. You can easily start training your agent using these verified implementations.  

For the list of implemented algorithms see [here](./nnabla_rl/algorithms/README.md).

You can also find the reproduction and evaluation results of each algorithm [here](./reproductions).  
Note that you may not get completely the same results when running the reproduction code on your computer. The result may slightly change depending on your machine, nnabla/nnabla-rl's package version, etc.

### Seemless switching of online and offline training

In reinforcement learning, there are two main training procedures, online and offline, to train the agent.
Online training is a training procedure that executes both data collection and network update alternately. Conversely, offline training is a training procedure that updates the network using only existing data. With NNablaRL, you can switch these two training procedures seemlessly. For example, as shown below, you can easily train a robot's controller online using simulated environment and finetune it offline with real robot dataset.

```py
import nnabla_rl
import nnabla_rl.algorithms as A

simulator = get_simulator() # This is just an example. Assuming that simulator exists
dqn = A.DQN(simulator)
# train online for 1M iterations
dqn.train_online(simulator, total_iterations=1000000)

real_data = get_real_robot_data() # This is also an example. Assuming that you have real robot data
# fine tune the agent offline for 10k iterations using real data
dqn.train_offline(real_data, total_iterations=10000)
```

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

Full documentation is [here](https://nnabla-rl.readthedocs.io/).

## Contribution guide

Any kind of contribution to NNablaRL is welcome! See the [contribution guide](./CONTRIBUTING.md) for details.

## License

NNablaRL is provided under the [Apache License Version 2.0](LICENSE) license.
