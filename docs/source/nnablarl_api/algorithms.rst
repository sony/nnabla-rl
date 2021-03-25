===========
Algorithms
===========
All algorithm are derived from :class:`nnabla_rl.algorithm.Algorithm`.

..  note::

   Algorithm will run on cpu by default 
   (No matter what `nnabla context <https://nnabla.readthedocs.io/en/latest/python/api/common.html#context>`_ 
   is set in prior to the instantiation). 
   If you want to run the algorithm on gpu, set the gpu_id through the algorithm's config.
   Note that the algorithm will override the nnabla context when the training starts.

Algorithm
==========
.. autoclass:: nnabla_rl.algorithm.AlgorithmConfig
   :members:

.. autoclass:: nnabla_rl.algorithm.Algorithm
   :members:

A2C
====
.. autoclass:: nnabla_rl.algorithms.a2c.A2CConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.a2c.A2C
   :members:
   :show-inheritance:


BCQ
====
.. autoclass:: nnabla_rl.algorithms.bcq.BCQConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.bcq.BCQ
   :members:
   :show-inheritance:

BEAR
=====
.. autoclass:: nnabla_rl.algorithms.bear.BEARConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.bear.BEAR
   :members:
   :show-inheritance:

Categorical DQN
================
.. autoclass:: nnabla_rl.algorithms.categorical_dqn.CategoricalDQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.categorical_dqn.CategoricalDQN
   :members:
   :show-inheritance:

DDPG
=====
.. autoclass:: nnabla_rl.algorithms.ddpg.DDPGConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.ddpg.DDPG
   :members:
   :show-inheritance:

DQN
====
.. autoclass:: nnabla_rl.algorithms.dqn.DQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.dqn.DQN
   :members:
   :show-inheritance:

GAIL
=====
.. autoclass:: nnabla_rl.algorithms.gail.GAILConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.gail.GAIL
   :members:
   :show-inheritance:

IQN
====
.. autoclass:: nnabla_rl.algorithms.iqn.IQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.iqn.IQN
   :members:
   :show-inheritance:

Munchausen DQN
===============
.. autoclass:: nnabla_rl.algorithms.munchausen_dqn.MunchausenDQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.munchausen_dqn.MunchausenDQN
   :members:
   :show-inheritance:


Munchausen IQN
===============
.. autoclass:: nnabla_rl.algorithms.munchausen_iqn.MunchausenIQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.munchausen_iqn.MunchausenIQN
   :members:
   :show-inheritance:

PPO
====
.. autoclass:: nnabla_rl.algorithms.ppo.PPOConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.ppo.PPO
   :members:
   :show-inheritance:

QRDQN
======
.. autoclass:: nnabla_rl.algorithms.qrdqn.QRDQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.qrdqn.QRDQN
   :members:
   :show-inheritance:

REINFORCE
==========
.. autoclass:: nnabla_rl.algorithms.reinforce.REINFORCEConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.reinforce.REINFORCE
   :members:
   :show-inheritance:

SAC
====
.. autoclass:: nnabla_rl.algorithms.sac.SACConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.sac.SAC
   :members:
   :show-inheritance:

SAC (ICML 2018 version)
=========================
.. autoclass:: nnabla_rl.algorithms.icml2018_sac.ICML2018SACConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.icml2018_sac.ICML2018SAC
   :members:
   :show-inheritance:

TD3
====
.. autoclass:: nnabla_rl.algorithms.td3.TD3Config
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.td3.TD3
   :members:
   :show-inheritance:

TRPO
=====
.. autoclass:: nnabla_rl.algorithms.trpo.TRPOConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.trpo.TRPO
   :members:
   :show-inheritance:

TRPO (ICML 2015 version)
==========================
.. autoclass:: nnabla_rl.algorithms.icml2015_trpo.ICML2015TRPOConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.icml2015_trpo.ICML2015TRPO
   :members:
   :show-inheritance:
