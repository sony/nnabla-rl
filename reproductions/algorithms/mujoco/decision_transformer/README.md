# Decision transformer reproduction

This reproduction script trains the Decision transformer algorithm proposed by proposed by L. Chen, et al. 
in the paper: [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

## Prerequisite

Install d4rl

```
$ pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

## How to run the reproduction script

To run the reproduction script do

```sh
$ python decision_transformer_reproduction.py <options>
```

If you omit options, the script will run on HalfCheetah-v3 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python decision_transformer_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example: run the script on cpu and train the agent with Walker2d-v3:
$ python decision_transformer_reproduction.py --env Walker2d-v3 --gpu -1
```

To check all available options type:

```sh
$ python decision_transformer_reproduction.py --help
```

To check the trained result do

```sh
$ python decision_transformer_reproduction.py --showcase --snapshot-dir <snapshot_dir> --render
```

```sh
# Example:
$ python decision_transformer_reproduction.py --showcase --snapshot-dir ./HalfCheetah-v3/seed-1/iteration-1/ --render
```

## Evaluation procedure

We tested our implementation with 3 MuJoCo tasks also used in the [original paper](https://arxiv.org/abs/2106.01345) using 3 different initial random seeds:

We evaluated the algorithm by running 10 trials after each epoch.

## Result

Mean and variance of expert normalized scores across 3 seeds are as follows.

|Env|nnabla-rl normalized score [%]|Reported normalized score [%]|
|:---|:---:|:---:|
|HalfCheetah-v3|42.7+/-1.4|42.6+/-0.1|
|Hopper-v3|52.3+/-7.0|67.6+/-1.0|
|Walker2d-v3|76.7+/-12.8|74.0+/-1.4|
