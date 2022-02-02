# ICML2018SAC (Soft Actor Critic with reward scaling) reproduction

This reproduction script trains the SAC (Soft Actor Critic) algorithm proposed by T. Haarnoja et al. in the paper: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290).

## Prerequisites

Install [delayed mujoco environment](../environment).

```sh
$ cd ../environment
$ pip install -e .
```

## How to run the reproduction script

To run the reproduction script do

```sh
$ python icml2018sac_reproduction.py <options>
```

If you omit options, the script will run on DelayedAnt-v1 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python icml2018sac_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example1: run the script on cpu and train the agent with DelayedHalfCheetah:
$ python icml2018sac_reproduction.py --env DelayedHalfCheetah-v1 --gpu -1
# Example2: run the script on gpu 1 and train the agent with DelayedWalker2d:
$ python icml2018sac_reproduction.py --env DelayedWalker2d-v1 --gpu 1
```

To check all available options type:

```sh
$ python icml2018sac_reproduction.py --help
```

To check the trained result do

```sh
$ python icml2018sac_reproduction.py --showcase --snapshot-dir <snapshot_dir> --render
```

```sh
# Example:
$ python icml2018sac_reproduction.py --showcase --snapshot-dir ./DelayedAnt-v1/seed-1/iteration-10000/ --render
```

## Evaluation

We tested our implementation with 4 Delayed MuJoCo environments as in the [mme sac paper](https://arxiv.org/abs/2106.10517) using 5 different initial random seeds:

- DelayedAnt-v1
- DelayedHalfCheetah-v1
- DelayedHopper-v1
- DelayedWalker2d-v1

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|
|DelayedAnt-v1|5121.65+/-325.60|3248.43+/-1454.48|
|DelayedHalfCheetah-v1|5342.76+/-2756.57|3742.33+/-3064.55|
|DelayedHopper-v1|3137.58+/-274.17|2175.31+/-1358.39|
|DelayedWalker2d-v1|4316.92+/-307.31|3220.92+/-1107.91|

## Learning curves

### DelayedAnt-v1

![DelayedAnt-v1 Result](reproduction_results/DelayedAnt-v1_results/result.png)

### DelayedHalfCheetah-v1

![DelayedHalfCheetah-v1 Result](reproduction_results/DelayedHalfCheetah-v1_results/result.png)

### DelayedHopper-v1

![DelayedHopper-v1 Result](reproduction_results/DelayedHopper-v1_results/result.png)

### DelayedWalker2d-v1

![DelayedWalker2d-v1 Result](reproduction_results/DelayedWalker2d-v1_results/result.png)
