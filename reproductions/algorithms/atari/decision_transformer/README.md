# Decision transformer reproduction

This reproduction script trains the Decision transformer algorithm proposed by proposed by L. Chen, et al. 
in the paper: [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

## Prerequisite

Download [dqn-replay dataset](https://research.google/tools/datasets/dqn-replay/) as follows. </br>

### Install gsutil

See this [official installation note](https://cloud.google.com/storage/docs/gsutil_install#install) of gsutil.

### Download the dataset

First, create datasets directory.

```
$ mkdir datasets
```

(NOTE: Replace <env_name> by Breakout if you want to train on Breakout)

```
$ gsutil -m cp -R gs://atari-replay-datasets/dqn/<env_name> datasets
```

### Check downloaded dataset

Run dataset_viewer to check the download was success

```
$ python dataset_viewer.py
```

## How to run the reproduction script

To run the reproduction script do

```sh
$ python decision_transformer_reproduction.py <options>
```

If you omit options, the script will run on BreakoutNoFrameskip-v4 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python decision_transformer_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example: run the script on cpu and train the agent with Pong:
$ python decision_transformer_reproduction.py --env PongNoFrameskip-v4 --gpu -1
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
$ python decision_transformer_reproduction.py --showcase --snapshot-dir ./BreakoutNoFrameskip-v4/seed-1/iteration-1/ --render
```

## Evaluation procedure

We tested our implementation with 4 Atari games also used in the [original paper](https://arxiv.org/abs/2106.01345) using 3 different initial random seeds:

We evaluated the algorithm by running 10 trials after each epoch.

## Result

Mean and variance of raw scores and gamer normalized scores across 3 seeds are as follows.

|Env|nnabla_rl best score (normalized score [%])|Reported score (normalized score [%])|Baseline gamer score|
|:---|:---:|:---:|:---:|
|BreakoutNoFrameskip-v4|92.1+/-75.4 (307.0+/-268.4)|76.9+/-27.3 (267.5+/-97.5)|30|
|PongNoFrameskip-v4|14.9+/-5.9 (99.1+/-39.1)|17.1+/-2.9 (106.1+/-8.1)|15|
|QbertNoFrameskip-v4|12774.2+/-4177.7 (94.9+/-31.1)|2215.8+/-1523.7 (15.4+/-11.4)|13455|
|SeaquestNoFrameskip-v4|1231.3+/-382.4 (2.9+/-0.9)|1129.3+/189.0 (2.5+/-0.4)|42055|
