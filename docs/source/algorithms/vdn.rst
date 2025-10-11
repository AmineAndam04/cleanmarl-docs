Value-Decomposition Networks For Cooperative Multi-Agent Learning
=================================================================

    - Paper link:  `VDN <https://arxiv.org/abs/1706.05296>`_ 

Quick facts:
    - VDN is an off-policy and value-based algorithm.
    - VDN works only for discrete actions.
    - Needs a common reward.
    - Additive factorization.

Key ideas:
    - VDN learns centralized action-value function :math:`Q^{tot}` decomposed into the same of individual :math:`Q_i` networks

    .. math::

        Q(\mathbf{o}, \mathbf{a}) = \sum_{i \in I} Q_i(o_i, a_i).


    - This factorization allows us to have decentralized policies.

    - :math:`Q_i` networks are refereed to as *utility networks* instead of action-value function, as they don't satisfy the Bellman equation. Instead,  :math:`Q^{tot}` is a true action-value function.

VDN is based on Q-learning and works with settings with common reward :math:`r`. Let's forget about the VDN for now and focus on how Q-learning can solve the cooperative MARL problem. There are two approaches we can use, each with its pros and cons.

The first approach is to use a single-agent RL algorithm. This consists of considering that there is **one central agent** who receives the joint observation :math:`\mathbf{o}_t` and its action is the joint action :math:`\mathbf{a}_t`. Then our goal is to estimate :math:`Q(\mathbf{o}_t,\mathbf{a}_t;\theta)` as in DQN. In this case, the loss to optimize is:  

.. math::

   L(\theta) =  (y_t - Q(\mathbf{o}_t, \mathbf{a}_t; \theta))^2 \tag{1}

where:

.. math::
    
    y_t = r_t + \gamma(1-done) \max_{\mathbf{a'}_t} Q(\mathbf{o}_{t+1}, \mathbf{a'}_t; \theta^{-})

- Pros: The loss function is backpropagated using the team reward which is strongly related to the input of the Q-network: the joint action :math:`a`.
- Cons: The Q-network takes as input the joint observation :math:`o` and outputs joint action :math:`a`. This can be very problematic as we have to deal with extremely large inputs. Additionally, the output size  grow exponentially with the number of agents.



The second approach is to use independent Q-learning. This means that each agent will train its own Q-learning algorithm relying only on its local observation :math:`o_i` and local actions :math:`a_i` . Therefor we have the following :math:`n` loss functions to optimize:


.. math::
    
    L_i(\theta) =  (y_i^t - Q_i(o_i^t, a_i^t; \theta))^2 \tag{3}

.. math::
    
     y_i^t =r_t + \gamma(1-done)\max_{a'_i} Q(o_i^{t+1}, a'_i; \theta^{-}) 


- Pros: The Q-networks are trained using local observations :math:`o_i` and individual actions :math:`a_i`, allowing us to train these networks more efficiently and avoid large inputs. It also becomes easier to deploy these networks.
- Cons: The individual Q-networks backpropagate a reward signal that is a consequences of the actions of the other agents rather than their own

The idea of the VDN  is to combine the two approaches. First, we want to train decentralized network: we rely on local observations :math:`o_i` and local actions :math:`a_i`, thus we train local Q-networks :math:`Q_i(o_i^{t+1}, a'_t; \theta)`. 

Second, we want to use the common reward to backpropagate through a loss that **aggregates the agents**. As this can be done only when working with a centralized Q-network :math:`Q(\mathbf{o}_t, \mathbf{a}_t; \theta)`, we make the following assumption :

.. math::
    
     Q(\mathbf{o}, \mathbf{a}; \theta) = \sum_{i \in I} Q_i(o_i, a_i; \theta) \tag{5}

and we use the following loss function: 

.. math::
    
     L(\theta) = \left( r_t + \gamma \max_{\mathbf{a'} \in A} Q(\mathbf{o}_{t+1}, \mathbf{a'}; \theta^{-}) - Q(\mathbf{o}_{t}, \mathbf{a}_t; \theta) \right)^2 \tag{6}


with

.. math::
    
     Q(\mathbf{o}_{t}, \mathbf{a}_t; \theta) = \sum_{i \in I} Q_i(o_i^t, a_i^t; \theta) \tag{7}

and

.. math::
    
     \max_{\mathbf{a'} \in A} Q(\mathbf{o}_{t+1}, \mathbf{a'}; \theta^{-}) = \sum_{i \in I} \max_{a'_i \in A_i} Q_i(o^{t+1}_i, a'_i;\theta) \tag{8}


We don't propagate each individual Q-network separately, but instead, we backpropagate through the sum of the individual Q-networks. 

It's important to note that :math:`Q(.; \theta)` is not an actual neural network. We only instiantiate the individual networks  :math:`Q_i(.; \theta)`. Another thing is we can can have seprate weights for each network :math:`\theta_i`, instead of :math:`\theta`. However, sharing weights among agents is a common practice in MARL. 


Pseudocode
----------

.. image:: ../_static/vdn_algorithm.svg
   :alt: Architecture diagram
   :width: 100%
   :align: center

Implementations
---------------

We implemented three variants of VDN:

- ``vdn.py``: VDN with single environment and MLP neural networks.
- ``vdn_multienvs.py``: VDN with parallel environments and MLP neural networks.
- ``vdn_lstm.py``: VDN with single environment and recurrent neural networks.

Additional details:

- **Replay buffer**: For MLP-based implementations, we store transitions ``(obs, actions,reward,done,next_obs,next_avail_action)``. We need to store the ``next_avail_action`` in order to accurately compute the TD targets as we need the action-value of the best available next action. For the RNN-based implementation, we store sequences of transitions ``(seq_obs, seq_actions,seq_reward,seq_done,seq_next_obs,seq_next_avail_action)`` , and during the training we use the first ``burn_in`` transitions to compute the hidden state ``h``, and use the remaining of the sequence to update the network.

Logging
-------

We record the following metrics:

- **rollout/ep_reward** : Mean episode reward during environment rollouts.
- **rollout/ep_length** : Mean episode length during rollouts.
- **rollout/epsilon** : Current exploration epsilon.
- **rollout/battle_won** (SMAClite only): Fraction of battle won by SMAC agents
- **train/loss** : Training loss at the current optimization step.
- **train/grads** : Magnitude of gradients of the VDN networks.
- **eval/ep_reward** : Mean episode reward during evaluation.
- **eval/std_ep_reward** : Standard deviation of episode rewards during evaluation.
- **eval/ep_length** : Mean episode length during evaluation.
- **eval/battle_won** ( SMAClite only): Fraction of battles won during evaluation episodes.



Documentation
-------------

.. py:class::  cleanmarl.vdn.Args(env_type="smaclite", env_name="3m", env_family="sisl", agent_ids=True,           buffer_size=10000, total_timesteps=1000000, gamma=0.99, learning_starts=5000, train_freq=5, optimizer="Adam", learning_rate=0.0005, batch_size=32, start_e=1, end_e=0.05, exploration_fraction=0.05, hidden_dim=64, num_layers=1, target_network_update_freq=5, polyak=0.005, normalize_reward=False, clip_gradients=5, log_every=10, eval_steps=5000, num_eval_ep=10, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)

    :param env_type: Type of the environment: ``smaclite``, ``pz`` for PettingZoo, ``lbf`` for Level-based Foraging.
    :type env_type: str

    :param env_name: Name of the environment (``3m``, ``simple_spread_v3`` ``Foraging-2s-10x10-4p-2f-v3`` ...)
    :type env_name: str

    :param env_family: Env family when using a PettingZoo environment (``sisl``, ``mpe`` ...)
    :type env_family: str

    :param agent_ids: Include agent IDs (one-hot vector) in observations
    :type agent_ids: bool

    :param buffer_size: The size of the replay buffer
    :type buffer_size: int

    :param total_timesteps: Total steps of the environment during the training
    :type total_timesteps: int

    :param gamma: Discount factor
    :type gamma: float

    :param learning_starts: Number of environment steps to initialize the replay buffer
    :type learning_starts: int

    :param train_freq: Train the network each ``train_freq`` step in the environment
    :type train_freq: int

    :param optimizer: The optimizer
    :type optimizer: str

    :param learning_rate: Learning rate
    :type learning_rate: float

    :param batch_size: Batch size
    :type batch_size: int

    :param start_e: The starting value of epsilon, for exploration
    :type start_e: float

    :param end_e: The end value of epsilon, for exploration
    :type end_e: float

    :param exploration_fraction: The fraction of ``total-timesteps`` it takes from to go from ``start_e`` to ``end_e``.
    :type exploration_fraction: float

    :param hidden_dim: Hidden dimension
    :type hidden_dim: int

    :param num_layers: Number of layers
    :type num_layers: int

    :param target_network_update_freq: Update the target network each ``target_network_update_freq`` step in the environment
    :type target_network_update_freq: int

    :param polyak: Polyak coefficient to update the target network
    :type polyak: float

    :param normalize_reward: Normalize the rewards if True
    :type normalize_reward: bool

    :param clip_gradients: ``0<`` for no gradients clipping and ``0>`` if clipping gradients at ``clip_gradients``
    :type clip_gradients: float

    :param log_every: Log rollout stats every ``log_every`` episode
    :type log_every: int

    :param eval_steps: Evaluate the policy each ``eval_steps`` step
    :type eval_steps: int

    :param num_eval_ep: Number of evaluation episodes
    :type num_eval_ep: int

    :param use_wnb: Logging to Weights & Biases if True
    :type use_wnb: bool

    :param wnb_project: Weights & Biases project name
    :type wnb_project: str

    :param wnb_entity: Weights & Biases entity name
    :type wnb_entity: str

    :param device: Device (``cpu``, ``gpu``, ``mps``) *We only support CPU training for now*
    :type device: str

    :param seed: Random seed
    :type seed: int



.. py:class:: cleanmarl.vdn_lstm.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, buffer_size=10000, seq_length=10, burn_in=7, total_timesteps=1000000, gamma=0.99, learning_starts=5000, train_freq=5, optimizer="Adam", learning_rate=0.0007, batch_size=32, start_e=1, end_e=0.05, exploration_fraction=0.01, hidden_dim=64, num_layers=1, normalize_reward=False, target_network_update_freq=1, polyak=0.005, log_every=10, clip_gradients=1, eval_steps=10000, num_eval_ep=10, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)

    :param seq_length: Length of the sequence to store in the buffer
    :type seq_length: int

    :param burn_in: Sequences to burn during batch updates
    :type burn_in: int



.. py:class:: cleanmarl.vdn_multienvs.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, num_envs=4, buffer_size=10000, total_timesteps=1000000, gamma=0.99, learning_starts=5000, train_freq=2, optimizer="Adam", learning_rate=0.0005, batch_size=16, clip_gradients=5, start_e=1, end_e=0.05, exploration_fraction=0.05, hidden_dim=64, num_layers=1, target_network_update_freq=1, polyak=0.005, log_every=10, normalize_reward=False, eval_steps=5000, num_eval_ep=5, use_wnb=False, wnb_project="", wnb_entity="", device="mps", seed=1)

    :param num_envs: Number of parallel environments
    :type num_envs: int
