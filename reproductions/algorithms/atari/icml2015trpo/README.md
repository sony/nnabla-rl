# TRPO (Trust Region Policy Optimization) reproduction

This reproduction script trains the TRPO (Trust Region Policy Optimization) algorithm proposed by J. Schulman et al. in the paper: [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477).

## Prerequisite

Install gym[atari] via pip and install atari ROMS following [here](https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/examples/python-rom-package).

```
$ pip install gym[atari]
```

## How to run the reproduction script

To run the reproduction script do

```sh
$ python icml2015trpo_reproduction.py <options>
```

If you omit options, the script will run on BreakoutNoFrameskip-v4 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python icml2015trpo_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example1: run the script on cpu and train the agent with Pong:
$ python icml2015trpo_reproduction.py --env PongNoFrameskip-v4 --gpu -1
# Example2: run the script on gpu 1 and train the agent with SpaceInvaders:
$ python icml2015trpo_reproduction.py --env SpaceInvadersNoFrameskip-v4 --gpu 1
```

To check all available options type:

```sh
$ python icml2015trpo_reproduction.py --help
```

To check the trained result do

```sh
$ python icml2015trpo_reproduction.py --showcase --snapshot-dir <snapshot_dir> --render
```

```sh
# Example:
$ python icml2015trpo_reproduction.py --showcase --snapshot-dir ./BreakoutNoFrameskip-v4/seed-1/iteration-100000/ --render
```

## Evaluation

We tested our implementation with 1 Atari game also used in the [original paper](https://arxiv.org/pdf/1502.05477.pdf).
Please note that this version of TRPO uses Single Path method to estimate Q value instead of Generalized Advantage Estimation (GAE).
(Training is slow. We recommend using TRPO with GAE.)

Following Atari game was tested with 3 different initial random seeds:

- Pong

There is no original implemention of this version TRPO, so some parameters that are not described in the paper are retrieved from https://github.com/openai/baselines/blob/master/baselines/trpo_mpi/defaults.py

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|
|Pong|19.467+/-1.231|**20.9**|

## Learning curves

### Pong

![Pong Result](reproduction_results/PongNoFrameskip-v4_results/result.png)

