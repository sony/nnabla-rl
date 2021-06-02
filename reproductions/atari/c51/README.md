# C51 (Categorical DQN with 51 atoms) reproduction

This reproduction script trains the C51 (Categorical DQN with 51 atoms) algorithm proposed by M. G. Bellemare, et al. in the paper: [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887).

## How to run the reproduction script

To run the reproduction script do

```sh
$ python c51_reproduction.py <options>
```

If you omit options, the script will run on BreakoutNoFrameskip-v4 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python c51_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example1: run the script on cpu and train the agent with Pong:
$ python c51_reproduction.py --env PongNoFrameskip-v4 --gpu -1
# Example2: run the script on gpu 1 and train the agent with SpaceInvaders:
$ python c51_reproduction.py --env SpaceInvadersNoFrameskip-v4 --gpu 1
```

To check all available options type:

```sh
$ python c51_reproduction.py --help
```

To check the trained result do

```sh
$ python c51_reproduction.py --showcase --snapshot-dir <snapshot_dir> --render
```

```sh
# Example:
$ python c51_reproduction.py --showcase --snapshot-dir ./BreakoutNoFrameskip-v4/seed-1/iteration-250000/ --render
```

## Evaluation

We tested our implementation with 5 Atari games also used in the [original paper](https://arxiv.org/pdf/1707.06887.pdf) using 3 differnt initial random seeds:

- Asterix
- Breakout
- Pong
- Qbert
- Seaquest

We evaluated the algorithm in following settings.

* In every 1M frames (250K steps), the mean reward is evaluated using the Q-Network parameter at that timestep. 
* The evaluation step lasts for 500K frames (125K steps) but the last episode that exceeeds 125K timesteps is not used for evaluation.
* epsilon is set to 0.001 (not greedy).

Mean evaluation score is the mean score among 3 seeds at each iteration.

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|
|AsterixNoFrameskip-v4|450286.364+/-342066.032|406211|
|BreakoutNoFrameskip-v4|623.75+/-208.36|748|
|PongNoFrameskip-v4|20.995+/-0.067|21.0|
|QbertNoFrameskip-v4|26228.958+/-2191.249|26387.5|
|SeaquestFrameskip-v4|142828.571+/-197219.983|266434|

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