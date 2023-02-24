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

ATRPO
======
.. autoclass:: nnabla_rl.algorithms.atrpo.ATRPOConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.atrpo.ATRPO
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

Categorical DDQN
================
.. autoclass:: nnabla_rl.algorithms.categorical_ddqn.CategoricalDDQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.categorical_ddqn.CategoricalDDQN
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

DDP
=====
.. autoclass:: nnabla_rl.algorithms.ddp.DDPConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.ddp.DDP
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

DDQN
=====
.. autoclass:: nnabla_rl.algorithms.ddqn.DDQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.ddqn.DDQN
   :members:
   :show-inheritance:

DecisionTransformer
=====
.. autoclass:: nnabla_rl.algorithms.decision_transformer.DecisionTransformerConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.decision_transformer.DecisionTransformer
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

DRQN
=====
.. autoclass:: nnabla_rl.algorithms.drqn.DRQNConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.drqn.DRQN
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

HER
====
.. autoclass:: nnabla_rl.algorithms.her.HERConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.her.HER
   :members:
   :show-inheritance:

iLQR
====
.. autoclass:: nnabla_rl.algorithms.ilqr.iLQRConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.ilqr.iLQR
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

LQR
====
.. autoclass:: nnabla_rl.algorithms.lqr.LQRConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.lqr.LQR
   :members:
   :show-inheritance:

MMESAC
===============
.. autoclass:: nnabla_rl.algorithms.mme_sac.MMESACConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.mme_sac.MMESAC
   :members:
   :show-inheritance:

MMESAC (Disentangled)
===============
.. autoclass:: nnabla_rl.algorithms.demme_sac.DEMMESACConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.demme_sac.DEMMESAC
   :members:
   :show-inheritance:

MPPI
===============
.. autoclass:: nnabla_rl.algorithms.mppi.MPPIConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.mppi.MPPI
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

QRSAC
===============
.. autoclass:: nnabla_rl.algorithms.qrsac.QRSACConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.qrsac.QRSAC
   
QtOpt (ICRA 2018 version)
=========================
.. autoclass:: nnabla_rl.algorithms.icra2018_qtopt.ICRA2018QtOpt
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.icra2018_qtopt.ICRA2018QtOpt
   :members:
   :show-inheritance:

Rainbow
==========
.. autoclass:: nnabla_rl.algorithms.rainbow.RainbowConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.rainbow.Rainbow
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

SAC-D
====
.. autoclass:: nnabla_rl.algorithms.sac.SACDConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.sacd.SACD
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

XQL (eXtreme Q-Learning)
=====
.. autoclass:: nnabla_rl.algorithms.xql.XQLConfig
   :members:
   :show-inheritance:

.. autoclass:: nnabla_rl.algorithms.xql.XQL
   :members:
   :show-inheritance:

