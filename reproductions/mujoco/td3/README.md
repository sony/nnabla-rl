# TD3 (Twin Delayed Deep Deterministic policy gradient) reproduction

We tested our implementation with 7(+1) MuJoCo environments also used in the [original paper](https://arxiv.org/pdf/1802.09477.pdf).

We tested our implementation with following MuJoCo environments using 3 different initial random seeds:

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
|Ant-v2|**5056.422+/-938.283**|4372.44+/-1000.33|
|HalfCheetah-v2|**10627.57+/-843.172**|9636.95+/-859.065|
|Hopper-v2|3050.033+/-843.191|**3564.07+/-114.74**|
|InvertedDoublePendulum-v2|**9359.346+/-0.622**|9337.47+/-14.96|
|InvertedPendulum-v2|1000.0+/-0.0|1000.0+/-0.0|
|Reacher-v2|**-3.274+/-1.405**|-3.60+/-0.56|
|Swimmer-v2|56.297+/-24.402|N/A|
|Walker2d-v2|**4948.24+/-318.22**|4682.82+/-539.64|

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
