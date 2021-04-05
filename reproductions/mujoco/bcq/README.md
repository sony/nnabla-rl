# BCQ (Batch Constrained Q-learning) reproduction

## Prerequisites

Install [d4rl](https://github.com/rail-berkeley/d4rl) before starting the training. (And also you'll need the mujoco software license to use mujoco)

```sh
$ pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

## Run training

Use d4rl env for reproduction (This reproduction code only supports mujoco environment provided in d4rl).

```sh
$ python bcq_reproduction.py --env='ant-expert-v0'
```

## Tested environments

We trained the algorithm with the following environments.

- ant-expert-v0
- halfcheetah-expert-v0
- hopper-expert-v0
- walker2d-expert-v0

## Reproduction results

Red horizontal line is the expert's score. </br>
Shaded region is the standard deviation of evaluation score. </br>
The tested environment is not completely the same as the original paper but the result is similar to that of imitation performance presented in the [BCQ paper](https://arxiv.org/pdf/1812.02900.pdf).

### ant-expert-v0

![ant-expert-v0 Result](reproduction_results/ant-expert-v0_results/result.png)

### halfcheetah-expert-v0

![HalfCheetah-expert-v0 Result](reproduction_results/halfcheetah-expert-v0_results/result.png)

### hopper-expert-v0

![hopper-expert-v0 Result](reproduction_results/hopper-expert-v0_results/result.png)

### walker2d-expert-v0

![walker2d-expert-v0 Result](reproduction_results/walker2d-expert-v0_results/result.png)
