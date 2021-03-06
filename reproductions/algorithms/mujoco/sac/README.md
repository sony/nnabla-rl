# SAC (Soft Actor Critic) reproduction

This reproduction script trains the SAC (Soft Actor Critic) algorithm proposed by T. Haarnoja et al. in the paper: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905).

## How to run the reproduction script

To run the reproduction script do

```sh
$ python sac_reproduction.py <options>
```

If you omit options, the script will run on Ant-v2 environment with gpu id 0.

You can change the training environment and gpu as follows

```sh
$ python sac_reproduction.py --env <env_name> --gpu <gpu_id>
```

```sh
# Example1: run the script on cpu and train the agent with HalfCheetah:
$ python sac_reproduction.py --env HalfCheetah-v2 --gpu -1
# Example2: run the script on gpu 1 and train the agent with Walker2d:
$ python sac_reproduction.py --env Walker2d-v2 --gpu 1
```

To check all available options type:

```sh
$ python sac_reproduction.py --help
```

To check the trained result do

```sh
$ python sac_reproduction.py --showcase --snapshot-dir <snapshot_dir> --render
```

```sh
# Example:
$ python sac_reproduction.py --showcase --snapshot-dir ./Ant-v2/seed-1/iteration-10000/ --render
```

## Evaluation

We tested our implementation with the following MuJoCo environments using 3 different initial random seeds:

- Ant-v2
- HalfCheetah-v2
- Hopper-v2
- InvertedDoublePendulum-v2
- InvertedPendulum-v2
- Reacher-v2
- Swimmer-v2
- Walker2d-v2

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|
|Ant-v2|6333.712+/-582.962|~5500|
|HalfCheetah-v2|15628.84+/-122.80|~15000|
|Hopper-v2|3572.477+/-22.762|~3300|
|InvertedDoublePendulum-v2|9359.66+/-0.27|N/A|
|InvertedPendulum-v2|1000.0+/-0.0|N/A|
|Reacher-v2|-2.98+/-1.42|N/A|
|Swimmer-v2|66.465+/-8.231|N/A|
|Walker2d-v2|5372.365+/-669.718|~6000|

## Learning curves

### Ant-v2

![Ant-v2 Result](reproduction_results/Ant-v2_results/result.png)

### HalfCheetah-v2

![HalfCheetah-v2 Result](reproduction_results/HalfCheetah-v2_results/result.png)

### Hopper-v2

![Hopper-v2 Result](reproduction_results/Hopper-v2_results/result.png)

### InvertedDoublePendulum-v2

![InvertedDoublePendulum-v2 Result](reproduction_results/InvertedDoublePendulum-v2_results/result.png)

### InvertedPendulum-v2

![InvertedPendulum-v2 Result](reproduction_results/InvertedPendulum-v2_results/result.png)

### Reacher-v2

![Reacher-v2 Result](reproduction_results/Reacher-v2_results/result.png)

### Swimmer-v2

![Swimmer-v2 Result](reproduction_results/Swimmer-v2_results/result.png)

### Walker2d-v2

![Walker2d-v2 Result](reproduction_results/Walker2d-v2_results/result.png)
