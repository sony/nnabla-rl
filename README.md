[![License](https://img.shields.io/github/license/sony/nnabla-rl)](LICENSE)
[![Build status](https://github.com/sony/nnabla-rl/workflows/Build%20nnabla-rl/badge.svg)](https://github.com/sony/nnabla-rl/actions)
[![Documentation Status](https://readthedocs.org/projects/nnabla-rl/badge/?version=latest)](https://nnabla-rl.readthedocs.io/en/latest/?badge=latest)
[![Doc style](https://img.shields.io/badge/%20style-google-3666d6.svg)](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# Deep Reinforcement Learning Library built on top of Neural Network Libraries

nnablaRL is a deep reinforcement learning library built on top of [Neural Network Libraries](https://github.com/sony/nnabla) 
that is intended to be used for research, development and production.

## Installation

Installing nnablaRL is easy!

```sh
$ pip install nnabla-rl
```

nnablaRL only supports Python version >= 3.8 and [nnabla](https://github.com/sony/nnabla) version >= 1.17.

### Enabling GPU accelaration (Optional)

nnablaRL algorithms run on CPU by default. To run the algorithm on GPU, first install [nnabla-ext-cuda](https://github.com/sony/nnabla-ext-cuda) as follows.
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

nnablaRL has friendly Python APIs which enables to start training with only 3 lines of python code.
(NOTE: Below code will run on CPU. See the above instruction to run on GPU.)

```py
import nnabla_rl.algorithms as A
from nnabla_rl.utils.reproductions import build_classic_control_env

# Prerequisite: 
# Run below to enable rendering!
# $ pip install nnabla-rl[render]
env = build_classic_control_env("Pendulum-v1", render=True) # 1
ddpg = A.DDPG(env, config=A.DDPGConfig(start_timesteps=200))  # 2
ddpg.train(env)  # 3
```

To get more details about nnablaRL, see [documentation](https://nnabla-rl.readthedocs.io/) and [examples](./examples).

### Many builtin algorithms

Most of famous/SOTA deep reinforcement learning algorithms, such as DQN, SAC, BCQ, GAIL, etc., are implemented in nnablaRL. Implemented algorithms are carefully tested and evaluated. You can easily start training your agent using these verified implementations.  

For the list of implemented algorithms see [here](./nnabla_rl/algorithms/README.md).

You can also find the reproduction and evaluation results of each algorithm [here](./reproductions).  
Note that you may not get completely the same results when running the reproduction code on your computer. The result may slightly change depending on your machine, nnabla/nnabla-rl's package version, etc.

### Seamless switching of online and offline training

In reinforcement learning, there are two main training procedures, online and offline, to train the agent.
Online training is a training procedure that executes both data collection and network update alternately. Conversely, offline training is a training procedure that updates the network using only existing data. With nnablaRL, you can switch these two training procedures seamlessly. For example, as shown below, you can easily train a robot's controller online using simulated environment and finetune it offline with real robot dataset.

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

### Visualization of training graph and training progress

nnablaRL supports visualization of training graphs and training progresses with [nnabla-browser](https://github.com/sony/nnabla-browser)!

```py
import gym

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils.evaluator import EpisodicEvaluator

# save training computational graph
training_graph_hook = H.TrainingGraphHook(outdir="test")

# evaluation hook with nnabla's Monitor
eval_env = gym.make("Pendulum-v0")
evaluator = EpisodicEvaluator(run_per_evaluation=10)
evaluation_hook = H.EvaluationHook(
    eval_env,
    evaluator,
    timing=10,
    writer=W.MonitorWriter(outdir="test", file_prefix='evaluation_result'),
)

env = gym.make("Pendulum-v0")
sac = A.SAC(env)
sac.set_hooks([training_graph_hook, evaluation_hook])

sac.train_online(env, total_iterations=100)
```

![training-graph-visualization](./docs/source/images/training-graph-visualization.png)

![training-status-visualization](./docs/source/images/training-status-visualization.png)

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

Any kind of contribution to nnablaRL is welcome! See the [contribution guide](./CONTRIBUTING.md) for details.

## License

nnablaRL is provided under the [Apache License Version 2.0](LICENSE) license.
