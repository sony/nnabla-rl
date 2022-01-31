# Munchausen IQN (Munchausen Implicit Quantile Network) reproduction

This reproduction script trains the M-IQN (Munchausen Implicit Quantile Networks) algorithm proposed by N. Vieillard et al. in the paper: [Munchausen Reinforcement Learning](https://arxiv.org/pdf/1806.06923.pdf).

## Prerequisite

Install gym[atari] via pip and install atari ROMS following [here](https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/examples/python-rom-package).

```
$ pip install gym[atari]
```

## How to run the reproduction script

To run the reproduction script do

```sh
$ python munchausen_iqn_reproduction.py <options>
```

If you omit options, the script will run on BreakoutNoFrameskip-v4 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python munchausen_iqn_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example1: run the script on cpu and train the agent with Pong:
$ python munchausen_iqn_reproduction.py --env PongNoFrameskip-v4 --gpu -1
# Example2: run the script on gpu 1 and train the agent with SpaceInvaders:
$ python munchausen_iqn_reproduction.py --env SpaceInvadersNoFrameskip-v4 --gpu 1
```

To check all available options type:

```sh
$ python munchausen_iqn_reproduction.py --help
```

To check the trained result do

```sh
$ python munchausen_iqn_reproduction.py --showcase --snapshot-dir <snapshot_dir> --render
```

```sh
# Example:
$ python munchausen_iqn_reproduction.py --showcase --snapshot-dir ./BreakoutNoFrameskip-v4/seed-1/iteration-250000/ --render
```

## Atari Evaluation

We tested our implementation with 5 Atari games also used in the [original paper](https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf) with 3 different initial random seeds:

- Asterix
- BreakOut
- Pong
- Qbert
- Seaquest

We evaluated the algorithm as follows.

 * In every 1M frames (250K steps), the mean reward is evaluated using the Q-Network parameter at that timestep.
 * The evaluation step lasts for 500K frames (125K steps) but the last episode that exceeeds 125K timesteps is not used for evaluation.
 * epsilon is set to 0.001 (not greedy).

Mean evaluation score is the mean score among 3 seeds at each iteration.

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|
|AsterixNoFrameskip-v4|535135.417+/-250835.596|49865|
|BreakoutNoFrameskip-v4|782.167+/-157.641|320|
|PongNoFrameskip-v4|20.987+/-0.114|19|
|QbertNoFrameskip-v4|26152.199+/-2989.044|14739|
|SeaquestFrameskip-v4|19862.857+/-30763.661|23885|

## Learning curves

### Asterix

![Asterix Result](./reproduction_results/AsterixNoFrameskip-v4_results/result.png)

### Breakout

![Breakout Result](./reproduction_results/BreakoutNoFrameskip-v4_results/result.png)

### Pong

![Pong Result](./reproduction_results/PongNoFrameskip-v4_results/result.png)

### Qbert

![Qbert Result](./reproduction_results/QbertNoFrameskip-v4_results/result.png)

### Seaquest

![Seaquest Result](./reproduction_results/SeaquestNoFrameskip-v4_results/result.png)
