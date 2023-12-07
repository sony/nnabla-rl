# Algorithm catalog

nnabla-rl offers various (deep) reinforcement learning and optimal control algorithms. See the list below for the implemented algorithms! </br>

## Reinforcement learning algorithms

- Online training: Training which is performed by interacting with the environment. You'll need to prepare an environment which is compatible with the [OpenAI gym's environment interface](https://gym.openai.com/docs/#environments).
- Offline(Batch) training: Training which is performed sorely from provided data. You'll need to prepare a dataset capsuled with the [ReplayBuffer](../replay_buffer.py).
- Continuous/Discrete action: If you are familiar with the training of deep neural nets, the action type's difference is similar to the difference of regression and classification. Continuous action is an action which consists of real value(s) (e.g. robot's motor torque). In contrast, discrete action is an action which can be labeled (e.g. UP, DOWN, RIGHT, LEFT). The choice of action type depends on the environment (problem) and applicable algorithm changes depending on the its action type.
- Hybrid action: Hybrid action is an environment that requires both discrete and continuous action in pairs.
- RNN layer support: Supports training of network models with recurrent layers.

|Algorithm|Online training|Offline(Batch) training|Continuous action|Discrete action|Hybrid action|RNN layer support|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
|[A2C](https://arxiv.org/abs/1602.01783)|:heavy_check_mark:|:x:|(We will support continuous action in the future)|:heavy_check_mark:|:x:|:x:|
|[ATRPO](https://arxiv.org/pdf/2106.07329)|:heavy_check_mark:|:x:|:heavy_check_mark:|(We will support discrete action in the future)|:x:|:x:|
|[BCQ](https://arxiv.org/abs/1812.02900)|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:x:|
|[BEAR](https://arxiv.org/abs/1906.00949)|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:x:|
|[Categorical DDQN](https://arxiv.org/abs/1710.02298)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[Categorical DQN](https://arxiv.org/abs/1707.06887)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[DDPG](https://arxiv.org/abs/1509.02971)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[DDQN](https://arxiv.org/abs/1509.06461)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[DecisionTransformer](https://arxiv.org/abs/2106.01345)|:x:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|
|[DEMME-SAC](https://arxiv.org/abs/2106.10517)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[DQN](https://www.nature.com/articles/nature14236)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[DRQN](https://arxiv.org/abs/1507.06527)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[GAIL](https://arxiv.org/abs/1606.03476)|:heavy_check_mark:|:x:|:heavy_check_mark:|(We will support discrete action in the future)|:x:|:x:|
|[HER](https://arxiv.org/abs/1707.06347)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[HyAR](https://openreview.net/pdf?id=64trBbOhdGU)|:heavy_check_mark:|:x:|:x:|:x:|:heavy_check_mark:|:x:|
|[IQN](https://arxiv.org/abs/1806.06923)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:<sup>*</sup>|
|[MME-SAC](https://arxiv.org/abs/2106.10517)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[M-DQN](https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[M-IQN](https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[PPO](https://arxiv.org/abs/1707.06347)|:heavy_check_mark:|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|
|[QRSAC](https://www.nature.com/articles/s41586-021-04357-7)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[QRDQN](https://arxiv.org/abs/1710.10044)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[QtOpt (ICRA 2018 version)](https://arxiv.org/pdf/1802.10264)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[Rainbow](https://arxiv.org/abs/1710.02298)|:heavy_check_mark:|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|:heavy_check_mark:|
|[REDQ](https://arxiv.org/abs/2101.05982)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)|:heavy_check_mark:|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|
|[SAC](https://arxiv.org/abs/1812.05905)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[SAC (ICML 2018 version)](https://arxiv.org/abs/1801.01290)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[SAC-D](https://arxiv.org/abs/2206.13901)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[TD3](https://arxiv.org/abs/1802.09477)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|
|[TRPO](https://arxiv.org/abs/1502.05477)|:heavy_check_mark:|:x:|:heavy_check_mark:|(We will support discrete action in the future)|:x:|:x:|
|[TRPO (ICML 2015 version)](https://arxiv.org/abs/1502.05477)|:heavy_check_mark:|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|
|[XQL](https://arxiv.org/abs/2301.02328)|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:heavy_check_mark:|

<sup>*</sup>May require special treatment to train with RNN layers.

## Optimal control algorithms

- Need training: Most of the optimal control algorithm does NOT require training to run the controller. Instead, you will need the dynamics model of the system and cost function of the task in prior to the execution of the algorithm. See the documentation of each algorithm for the detail.
- Continuous/Discrete action: Same as reinfocement learning. However, most of the optimal control algorithm does not support discrete action.

|Algorithm|Need training|Continuous action|Discrete action|
|:---|:---:|:---:|:---:|
|[DDP](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf)|not required|:heavy_check_mark:|:x:|
|[iLQR](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf)|not required|:heavy_check_mark:|:x:|
|[LQR](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator)|not required|:heavy_check_mark:|:x:|
|[MPPI](https://ieeexplore.ieee.org/document/7989202)|may train a dynamics model|:heavy_check_mark:|:x:|
