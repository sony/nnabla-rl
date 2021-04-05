# A2C (Advantage Actor-Critic) reproduction

We tested our implementation with 5 Atari games also used in the [original paper](https://arxiv.org/abs/1602.01783).

Following Atari games was tested with 3 seeds:

- Asterix
- BreakOut
- Pong
- Qbert
- Seaquest

## Evaluation

We evaluated the algorithm in following settings.

* In every 1M frames (250K steps), the mean reward is evaluated using the Q-Network parameter at that timestep. 
* The evaluation step lasts for 500K frames (125K steps) but the last episode that exceeeds 125K timesteps is not used for evaluation.

All seeds results are combined and the mean of the score is calculated from them.

## Result

|Env|nnabla_rl best mean score|Reported score ([Reference](https://arxiv.org/pdf/1708.05144.pdf))|
|:---|:---:|:---:|
|AsterixNoFrameskip-v4|6678.691+/-3181.892|N/A|
|BreakoutNoFrameskip-v4|415.315+/-89.831|581.6|
|PongNoFrameskip-v4|20.335+/-1.11|19.9|
|QbertNoFrameskip-v4|15062.959+/-2643.344|15967.4|
|SeaquestFrameskip-v4|1743.03+/-81.934|1754.0|

**NOTE: Our A2C was trained with 8 threads.**

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