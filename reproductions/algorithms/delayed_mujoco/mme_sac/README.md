# MMESAC (Max-Min Entropy Soft Actor Critic) reproduction

This reproduction script trains the MME-SAC (Max-Min Entropy Soft Actor Critic) algorithm proposed by 
S. Han, et al. in the paper: [A Max-Min Entropy Framework for Reinforcement Learning](https://arxiv.org/abs/2106.10517)

## Prerequisites

Install [delayed mujoco environment](../environment).

```sh
$ cd ../environment
$ pip install -e .
```

## How to run the reproduction script

To run the reproduction script do

```sh
$ python mme_sac_reproduction.py <options>
```

If you omit options, the script will run on DelayedAnt-v1 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python mme_sac_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example1: run the script on cpu and train the agent with DelayedHalfCheetah:
$ python mme_sac_reproduction.py --env DelayedHalfCheetah-v1 --gpu -1
# Example2: run the script on gpu 1 and train the agent with DelayedWalker2d:
$ python mme_sac_reproduction.py --env DelayedWalker2d-v1 --gpu 1
```

To check all available options type:

```sh
$ python mme_sac_reproduction.py --help
```

To check the trained result do

```sh
$ python mme_sac_reproduction.py --showcase --snapshot-dir <snapshot_dir> --render
```

```sh
# Example:
$ python mme_sac_reproduction.py --showcase --snapshot-dir ./DelayedAnt-v1/seed-1/iteration-10000/ --render
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
|DelayedAnt-v1|5174.32+/-224.90|4664.04+/-836.37|
|DelayedHalfCheetah-v1|7036.85+/-2458.25|7299.28+/-1562.19|
|DelayedHopper-v1|3060.10+/-611.94|3421.32+/-88.29|
|DelayedWalker2d-v1|4724.53+/-504.59|5148.58+/-193.78|

## Learning curves

### DelayedAnt-v1

![DelayedAnt-v1 Result](reproduction_results/DelayedAnt-v1_results/result.png)

### DelayedHalfCheetah-v1

![DelayedHalfCheetah-v1 Result](reproduction_results/DelayedHalfCheetah-v1_results/result.png)

### DelayedHopper-v1

![DelayedHopper-v1 Result](reproduction_results/DelayedHopper-v1_results/result.png)

### DelayedWalker2d-v1

![DelayedWalker2d-v1 Result](reproduction_results/DelayedWalker2d-v1_results/result.png)
