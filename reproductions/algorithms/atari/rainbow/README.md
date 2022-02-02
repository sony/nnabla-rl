# Rainbow reproduction

This reproduction script trains the Rainbow algorithm proposed by M. G. Bellemare, et al. in the paper: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298).

## Prerequisite

Install gym[atari] via pip and install atari ROMS following [here](https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/examples/python-rom-package).

```
$ pip install gym[atari]
```

## How to run the reproduction script

To run the reproduction script do

```sh
$ python rainbow_reproduction.py <options>
```

If you omit options, the script will run on BreakoutNoFrameskip-v4 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python rainbow_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example1: run the script on cpu and train the agent with Pong:
$ python rainbow_reproduction.py --env PongNoFrameskip-v4 --gpu -1
# Example2: run the script on gpu 1 and train the agent with SpaceInvaders:
$ python rainbow_reproduction.py --env SpaceInvadersNoFrameskip-v4 --gpu 1
```

To check all available options type:

```sh
$ python rainbow_reproduction.py --help
```

To check the trained result do

```sh
$ python rainbow_reproduction.py --showcase --snapshot-dir <snapshot_dir> --render
```

```sh
# Example:
$ python rainbow_reproduction.py --showcase --snapshot-dir ./BreakoutNoFrameskip-v4/seed-1/iteration-250000/ --render
```

## Evaluation

We tested our implementation with 5 Atari games also used in the [original paper](https://arxiv.org/pdf/1710.02298.pdf) using 3 different initial random seeds:

- Asterix
- Breakout
- Pong
- Qbert
- Seaquest

We evaluated the algorithm in following settings.

* In every 1M frames (250K steps), the mean reward is evaluated using the Q-Network parameter at that timestep.
* The evaluation step lasts for 500K frames (125K steps) but the last episode that exceeeds 125K timesteps is not used for evaluation.
* Environment's timelimit was set to 108k frames (30 min).

Mean evaluation score is the mean score among 3 seeds.

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|
|AsterixNoFrameskip-v4|**555805.556+/-438292.712**|428200.3|
|BreakoutNoFrameskip-v4|368.0+/-62.53|**417.5**|
|PongNoFrameskip-v4|**21.0+/-0.0**|20.9|
|QbertNoFrameskip-v4|**39380.515+/-5682.445**|33817.5|
|SeaquestFrameskip-v4|6771.182+/-5043.452|**15898.9**|

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

## Ablation study results

We also conducted an ablation study. The results are the following.  
We only conducted the experiment with one seed.  
Learning curves are smoothed with a moving average of 10.

### Asterix

![Asterix Result](./ablation_study/AsterixNoFrameskip-v4/result.png)

### Breakout

![Breakout Result](./ablation_study/BreakoutNoFrameskip-v4/result.png)

### Pong

![Pong Result](./ablation_study/PongNoFrameskip-v4/result.png)

### Qbert

![Qbert Result](./ablation_study/QbertNoFrameskip-v4/result.png)

### Seaquest

![Seaquest Result](./ablation_study/SeaquestNoFrameskip-v4/result.png)
