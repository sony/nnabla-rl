# ICML2018SAC (Soft Actor Critic with reward scaling) reproduction

We tested our implementation with 4 MuJoCo environments as in the [original paper](https://arxiv.org/pdf/1801.01290.pdf).

We tested our implementation with following MuJoCo environments using 3 different initial random seeds:

- Ant-v2
- HalfCheetah-v2
- Hopper-v2
- Walker2d-v2

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|
|Ant-v2|6040.918+/-549.097|~6000|
|HalfCheetah-v2|14708.591+/-909.247|~15000|
|Hopper-v2|3388.758+/-135.154|~3300|
|Walker2d-v2|3814.35+/-233.07|~3800|

## Learning curves

### Ant-v2

![Ant-v2 Result](reproduction_results/Ant-v2_results/result.png)

### HalfCheetah-v2

![HalfCheetah-v2 Result](reproduction_results/HalfCheetah-v2_results/result.png)

### Hopper-v2

![Hopper-v2 Result](reproduction_results/Hopper-v2_results/result.png)

### Walker2d-v2

![Walker2d-v2 Result](reproduction_results/Walker2d-v2_results/result.png)
