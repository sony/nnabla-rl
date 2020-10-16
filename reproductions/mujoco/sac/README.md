# SAC (Soft Actor Critic) reproduction

We tested our implementation with 4 MuJoCo environments as in the [original paper](https://arxiv.org/pdf/1812.05905.pdf). </br>
We tested the algorithm of learned temperature version.

Following MuJoCo environments was tested with single seed:

- Ant-v2
- HalfCheetah-v2
- Hopper-v2
- Walker2d-v2

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|:---:|
|Ant-v2|6623.038+/-73.409|~5500|
|HalfCheetah-v2|15053.302+/-75.727|~15000|
|Hopper-v2|3631.186+/-2.685|~3300|
|Walker2d-v2|5503.636+/-8.55|~6000|

## Learning curves

### Ant-v2

![Ant-v2 Result](reproduction_results/Ant-v2_results/result.png)

### HalfCheetah-v2

![HalfCheetah-v2 Result](reproduction_results/HalfCheetah-v2_results/result.png)

### Hopper-v2

![Hopper-v2 Result](reproduction_results/Hopper-v2_results/result.png)

### Walker2d-v2

![Walker2d-v2 Result](reproduction_results/Walker2d-v2_results/result.png)
