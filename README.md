# Reinforcement Learning Libraries built on top of Neural Network Libraries

NNablaRL is a deep reinforcement learning framework that is intended to be used for research, development and production.

## Installation (tentative)

Installing NNablaRL is easy:

```sh
pip install -e .
```

NNablaRL only supports Python version 3.6 or greater.

## Features

### Friendly API

NNablaRL has friendly Python APIs which enables to start training with only 3 lines of python code.

```py
import nnabla_rl
import nnabla_rl.algorithms as A
from nnabla_rl.utils.reproductions import build_atari_env

env = build_atari_env("EnvName") # 1
dqn = A.DQN(env)  # 2
dqn.train()  # 3
```

To get more details about NNablaRL, see documentation and [examples](./examples).

### Builtin Algorithms

See [algorithms](./nnabla_rl/algorithms).

## Documentation

TBD.

## Contribution guide

TBD.

## License

TBD.