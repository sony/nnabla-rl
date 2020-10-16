# TD3 (Twin Delayed Deep Deterministic policy gradient) reproduction

We tested our implementation with 4 MuJoCo environments also used in the [original paper](https://arxiv.org/pdf/1802.09477.pdf).

Following MuJoCo environments was tested with single seed:

- Ant-v2
- HalfCheetah-v2
- Hopper-v2
- Walker2d-v2

## Result

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|:---:|
|Ant-v2|**5946.96+/-129.8**|4372.44+/-1000.33|
|HalfCheetah-v2|**10704.76+/-106.46**|9636.95+/-859.065|
|Hopper-v2|**3725.68+/-13.086**|3564.07+/-114.74|
|Walker2d-v2|**5399.54+/-54.353**|4682.82+/-539.64|

## Learning curves

### Ant-v2

![Ant-v2 Result](reproduction_results/Ant-v2_results/result.png)

### HalfCheetah-v2

![HalfCheetah-v2 Result](reproduction_results/HalfCheetah-v2_results/result.png)

### Hopper-v2

![Hopper-v2 Result](reproduction_results/Hopper-v2_results/result.png)

### Walker2d-v2

![Walker2d-v2 Result](reproduction_results/Walker2d-v2_results/result.png)
