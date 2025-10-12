Monotonic Value Function Factorization
======================================

    - Paper link:  `QMIX <https://arxiv.org/abs/1803.11485>`_ 

Quick facts:
    - QMIX is an off-policy and value-based algorithm.
    - QMIX works only with discrete actions.

Background
----------

VDN allows us to derive decentralized policies from a centralized action-value function.  
However, it has two major limitations that QMIX addresses:

1. The additive assumption is too restrictive.  VDN limits the representational capacity of the centralized value function by enforcing a simple sum of individual Q-values, rather than allowing a more flexible non-linear combination.

2. VDN cannot exploit additional information from the global state when available.


Both VDN and QMIX share the same objective: to extract decentralized policies from a centralized action-value function.  
While VDN supports only **linear decomposition**, QMIX supports **complex, non-linear decompositions**.


The key idea of QMIX is that decentralized policies can be extracted from a centralized network, as long as a **consistency property** is satisfied.  
Specifically, the global argmax over the centralized action-value function :math:`Q^{tot}` should be equivalent to performing individual argmax operations on each local value function :math:`Q_i`:

.. math::
   :nowrap:
   :label: eq-loss

   \begin{equation}
   \arg\max_{a} Q^{\text{tot}}(\mathbf{s}, \mathbf{o},\mathbf{a})
   =
   \begin{pmatrix}
   \arg\max_{a_1} Q_1(o_1, a_1) \\
   \vdots \\
   \arg\max_{a_n} Q_n(o_n, a_n)
   \end{pmatrix}
   \end{equation}


Our goal is to find a **mixing function** :math:`g` that satisfies :eq:`eq-loss`, such that:


.. math::
    Q^{\text{tot}}(\mathbf{s}, \mathbf{o},\mathbf{a}) = g(\mathbf{s}, Q_1(o_1, a_1;\theta), \dots,Q_n(o_n, a_n;\theta); \phi)

It is worth noting that VDN already satisfy this property. 

A **sufficient (but not necessary)** condition for a function :math:`g` to satisfy this property is to enforce **monotonicity** of :math:`Q^{tot}` whit respect to :math:`Q_i`:

.. math::

   \frac{\partial Q^{\text{tot}}}{\partial Q_i} \ge 0, \quad \forall i \in \mathcal{I}

In general, for a neural network :math:`g(\cdot; \phi)` to be monotonic with respect to its inputs, all its weights must be **non-negative**.  
QMIX uses this property to design a monotonic mixing function.

To achieve this, QMIX uses three neural networks:

- **Individual action-value networks:** :math:`Q_i(o_i, a_i)`
- **A hypernetwork**, which takes the global state :math:`s` as input and generates a set of **positive weights** :math:`\phi`. These weights parameterize the mixing network.
- **A mixing network**, whose parameters are produced by the hypernetwork, takes the individual Q-values as input and outputs the centralized value :math:`Q^{tot}`.


.. image:: ../_static/qmix_network.png
   :alt: Architecture diagram
   :width: 700px
   :align: center



Finally, the networks are trained by minimizing the following TD loss:

.. math::

    r + \gamma (1- done) \max_{\mathbf{a'}} Q^{tot}(\mathbf{s'},\mathbf{o'},\mathbf{a'}; \theta^-, \phi^-) - Q^{tot}(\mathbf{s},\mathbf{o},\mathbf{a}; \theta, \phi)




Pseudocode
----------

.. image:: ../_static/qmix_algorithm.svg
   :alt: Architecture diagram
   :width: 100%
   :align: center


Implementations
---------------

We implemented four variants of QMIX:

- ``qmix.py``: QMIX with a single environment and MLP neural networks.
- ``qmix_memefficient.py``: QMIX with a single environments and MLP neural networks, but with a memory-efficient replay buffer.
- ``qmix_multienvs.py``: QMIX with parallel environments and MLP neural networks.
- ``qmix_lstm.py``: QMIX with single environment and recurrent neural networks.

Additional details:

- **Replay buffer**: The replay buffer stores **episodes** instead of individual transitions. Therefore, we sample **batches of episodes** rather than batches of transitions. Each episode is initially stored as a dictionary with the following keys (except in ``qmix_memefficient.py``): ``{"obs": [], "actions": [], "reward": [], "next_obs": [], "states": [], "next_states": [], "done": [], "next_avail_actions": []}`` . This is not memory-efficient. For example, the observation at ``t=1`` is stored twice, once as ``obs`` and once as ``next_obs``.  A more memory-efficient strategy is implemented in ``qmix_memefficient.py``, where each episode is stored as: ``{"obs": [], "actions": [], "reward": [], "states": [], "done": [], "next_avail_actions": []}`` . We  need to store ``next_avail_actions`` to correctly compute TD targets, since the TD update requires the value of the best available next action.

- **Parallel environments**: Parallel environments are less critical for off-policy algorithms than for on-policy settings, since training samples are drawn from a replay buffer. To maintain a consistent number of network updates, we perform **multiple epochs per training step**, configurable with the ``n_epochs`` argument. The total number of network updates is logged under ``train/num_updates``.

- **RNN training** : We use **Truncated Backpropagation Through Time (TBPTT)** to train the RNN network. You can set the length of the sequence using ``tbptt``. 

Logging
-------

We record the following metrics:

- **rollout/ep_reward** : Mean episode reward during environment rollouts.
- **rollout/ep_length** : Mean episode length during rollouts.
- **rollout/epsilon** : Current exploration epsilon.
- **rollout/num_episodes** : Total number of completed episodes until the current step.
- **rollout/battle_won** (SMAClite only): Fraction of battle won by SMAC agents
- **train/loss** : Training loss at the current optimization step.
- **train/grads** : Magnitude of gradients of the VDN networks.
- **train/num_updates** : Total number of network updates until the current step.
- **eval/ep_reward** : Mean episode reward during evaluation.
- **eval/std_ep_reward** : Standard deviation of episode rewards during evaluation.
- **eval/ep_length** : Mean episode length during evaluation.
- **eval/battle_won** ( SMAClite only): Fraction of battles won during evaluation episodes.

Documentation
-------------

.. py:class:: cleanmarl.qmix.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, buffer_size=5000, total_timesteps=1000000, gamma=0.99, train_freq=1, optimizer="Adam", learning_rate=0.0005, batch_size=10, start_e=1, end_e=0.025, exploration_fraction=0.05, hidden_dim=64, hyper_dim=64, num_layers=1, target_network_update_freq=1, polyak=0.01, normalize_reward=False, clip_gradients=-1, log_every=10, eval_steps=50, num_eval_ep=5, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)

    :param env_type: Type of the environment: ``smaclite``, ``pz`` for PettingZoo, ``lbf`` for Level-based Foraging.
    :type env_type: str

    :param env_name: Name of the environment (``3m``, ``simple_spread_v3`` ``Foraging-2s-10x10-4p-2f-v3`` ...)
    :type env_name: str

    :param env_family: Env family when using a PettingZoo environment (``sisl``, ``mpe`` ...)
    :type env_family: str

    :param agent_ids: Include agent IDs (one-hot vector) in observations
    :type agent_ids: bool

    :param buffer_size: The number of episodes in the replay buffer
    :type buffer_size: int

    :param total_timesteps: Total steps in the environment during training
    :type total_timesteps: int

    :param gamma: Discount factor
    :type gamma: float

    :param train_freq: Train the network each ``train_fre`` episodes of the environment
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

    :param exploration_fraction: The fraction of ``total-timesteps`` it takes from to go from ``start_e`` to ``end_e``
    :type exploration_fraction: float

    :param hidden_dim: Hidden dimension of :math:`Q_i`:
    :type hidden_dim: int

    :param hyper_dim: Hidden dimension of the hyper-network
    :type hyper_dim: int

    :param num_layers: Number of layers
    :type num_layers: int

    :param target_network_update_freq: Update the target network each ``target_network_update_freq`` step in the environment
    :type target_network_update_freq: int

    :param polyak: Polyak coefficient when using polyak averaging for target network update
    :type polyak: float

    :param normalize_reward: Normalize the rewards if True
    :type normalize_reward: bool

    :param clip_gradients: ``0<`` for no gradients clipping and ``0>`` if clipping gradients at ``clip_gradients``
    :type clip_gradients: float

    :param log_every: Log rollout stats every ``log_every`` episode
    :type log_every: int

    :param eval_steps: Evaluate the policy each ``eval_steps`` episode
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

.. py:class:: cleanmarl.qmix_memefficient.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, buffer_size=5000, total_timesteps=1000000, gamma=0.99, train_freq=1, optimizer="Adam", learning_rate=0.0005, batch_size=10, start_e=1, end_e=0.025, exploration_fraction=0.05, hidden_dim=64, hyper_dim=64, num_layers=1, target_network_update_freq=1, polyak=0.01, normalize_reward=False, clip_gradients=-1, log_every=10, eval_steps=50, num_eval_ep=5, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)


.. py:class:: cleanmarl.qmix_multienvs.Args(env_type="smaclite", env_name="MMM", env_family="mpe", num_envs=4, agent_ids=True, buffer_size=5000, total_timesteps=1000000, gamma=0.99, train_freq=2, optimizer="Adam", learning_rate=0.0005, batch_size=32, start_e=1, end_e=0.025, exploration_fraction=0.05, hidden_dim=64, hyper_dim=64, num_layers=1, target_network_update_freq=1, polyak=0.005, clip_gradients=-1, n_epochs=2, normalize_reward=False, log_every=10, eval_steps=50, num_eval_ep=5, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)

    :param num_envs: Number of parallel environments
    :type num_envs: int

    :param n_epochs: Number of batches sampled in one update
    :type n_epochs: int


.. py:class:: cleanmarl.qmix_lstm.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, buffer_size=10000, total_timesteps=1000000, gamma=0.99, train_freq=1, optimizer="Adam", learning_rate=0.0008, batch_size=10, start_e=1, end_e=0.025, exploration_fraction=0.05, hidden_dim=64, hyper_dim=64, num_layers=1, target_network_update_freq=1, polyak=0.005, normalize_reward=False, clip_gradients=-1, tbptt=10, log_every=10, eval_steps=50, num_eval_ep=10, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)

    :param tbptt: Chunk size for Truncated Backpropagation Through Time (TBPTT).
    :type tbptt: int
