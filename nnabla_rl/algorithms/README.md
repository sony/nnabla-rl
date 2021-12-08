# Algorithm catalog

- Online training: Training which is performed by interacting with the environment. You'll need to prepare an environment which is compatible with the [OpenAI gym's environment interface](https://gym.openai.com/docs/#environments).
- Offline(Batch) training: Training which is performed sorely from provided data. You'll need to prepare a dataset capsuled with the [ReplayBuffer](../replay_buffer.py).
- Continuous/Discrete action: If you are familiar with the training of deep neural nets, the action type's difference is similar to the difference of regression and classification. Continuous action is an action which consists of real value(s) (e.g. robot's motor torque). In contrast, discrete action is an action which can be labeled (e.g. UP, DOWN, RIGHT, LEFT). The choice of action type depends on the environment (problem) and applicable algorithm changes depending on the its action type.

|Algorithm|Online training|Offline(Batch) training|Continuous action|Discrete action|
|:---|:---:|:---:|:---:|:---:|
|[A2C](https://arxiv.org/abs/1602.01783)|:heavy_check_mark:|:x:|(We will support continuous action in the future)|:heavy_check_mark:|
|[ATRPO](https://arxiv.org/pdf/2106.07329)|:heavy_check_mark:|:x:|:heavy_check_mark:|(We will support discrete action in the future)|
|[BCQ](https://arxiv.org/abs/1812.02900)|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|
|[BEAR](https://arxiv.org/abs/1906.00949)|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|
|[Categorical DDQN](https://arxiv.org/abs/1710.02298)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[Categorical DQN](https://arxiv.org/abs/1707.06887)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[DDPG](https://arxiv.org/abs/1509.02971)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|
|[DDQN](https://arxiv.org/abs/1509.06461)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[DQN](https://www.nature.com/articles/nature14236)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[GAIL](https://arxiv.org/abs/1606.03476)|:heavy_check_mark:|:x:|:heavy_check_mark:|(We will support discrete action in the future)|
|[M-DQN](https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[M-IQN](https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[PPO](https://arxiv.org/abs/1707.06347)|:heavy_check_mark:|:x:|:heavy_check_mark:|:heavy_check_mark:|
|[QRDQN](https://arxiv.org/abs/1710.10044)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[Rainbow](https://arxiv.org/abs/1710.02298)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)|:heavy_check_mark:|:x:|:heavy_check_mark:|:heavy_check_mark:|
|[SAC](https://arxiv.org/abs/1812.05905)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|
|[SAC (ICML 2018 version)](https://arxiv.org/abs/1801.01290)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|
|[TD3](https://arxiv.org/abs/1802.09477)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|
|[TRPO](https://arxiv.org/abs/1502.05477)|:heavy_check_mark:|:x:|:heavy_check_mark:|(We will support discrete action in the future)|
|[TRPO (ICML 2015 version)](https://arxiv.org/abs/1502.05477)|:heavy_check_mark:|:x:|:heavy_check_mark:|:heavy_check_mark:|

**<sup>*</sup>We do NOT support recurrent network structure (as of version 0.10)**