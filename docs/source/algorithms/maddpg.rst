Multi-Agent Deep Deterministic Policy Gradient
==============================================


    - Paper link:  `MADDPG <https://arxiv.org/abs/1706.02275>`_ 

Quick facts:
    - MADDPG is an off-policy actor-critic algorithm.
    - MADDPG uses a centralized critic with decentralized actors.
    - MADDPG support continuous and discrete actions. 
    - MADDPG support individual rewards, thus can be used for cooperative, competitive, and mixed. 

Background
----------
 

MADDPG is an extension of DDPG to multi-agent settings, using a centralized critic. The critic network takes as input the state information and all agents' actions :math:`Q(\mathbf{s},\mathbf{a};\phi)` and outputs a single value. When using individual rewards, separate centralized critics can be used: :math:`Q_i(\mathbf{s},\mathbf{a};\phi_i)`. Each agent has a deterministic policy :math:`\mu(o_i;\theta)`. For discrete actions, we use Gumbel-Softmax to compute gradients with respect to actions.


MADDPG is off-policy, therefor we store transitions while interacting with the environment and sample batches for training the actor and critic. Superscript :math:`b` indicates values sampled from the replay buffer.

To train the critic we use the following loss:

.. math::
    r + \gamma Q(\mathbf{s'^b},\mu(o'^b_1;\theta^-), \dots , \mu(o'^b_n;\theta^-); \phi^-) - Q(\mathbf{s}^b,a^b_1, \dots , a^b_n; \phi)


To better understand how to update the actor of agent :math:`i` , it's better if we pay close attention to the used gradient:

.. math::

   \nabla_{\theta} \mu_i(o_i) \, \nabla_{a_i} Q(s, a^b_1, \dots,a_i =\mu_i(o_i), \dots , a^b_n) 


When computing the gradient of agent :math:`i`, all the actions are from the replay buffer, except the :math:`i`-th action, which is replaced by the current policy.

.. image:: ../_static/maddpg_network.png
   :alt: Architecture diagram
   :width: 500px
   :align: center


Pseudocode
----------

.. image:: ../_static/maddpg_algorithm.svg
   :alt: Architecture diagram
   :width: 100%
   :align: center

Implementations
---------------

We implemented four variants of MADDPG:

- ``maddpg.py``: MADDPG with a single environment and MLP neural networks.
- ``maddpg_multienvs.py``: MADDPG with parallel environments and MLP neural networks.
- ``maddpg_lstm.py``: MADDPG with single environment and recurrent neural networks.
- ``maddpg_lstm_multienvs.py``: MADDPG with parallel environments and recurrent neural networks.

Additional details:

- **Replay buffer**: The replay buffer stores episodes instead of transitions, therefore, we sample batch of episodes rather than a batch of transitions. Each episode is stored as ``{"obs": [],"actions":[],"reward":[],"states":[],"done":[],"next_avail_actions":[]}`` . We store ``next_avail_actions`` to accurately compute TD targets for the best available next action.
- **Discrete actions**: we only support discrete actions for now
- **Gumbel-Softmax**: We use PyTorch's ``torch.nn.functional.gumbel_softmax``. During episode collection and critic training, set ``hard=True``; during actor training, set ``hard=False`` for better results. 
- **Parallel environments**: Parallel environments are not as useful for off-policy algorithms as for on-policy settings as we sample from a replay buffer. In order to keep comparable number of network updates, we train for multiple epochs in each training step by adding a ``n_epochs`` argument. We log the number of network updates under the name ``train/num_updates``. 
- **Parallel environments with RNNs**: When using multiple environments in parallel, some episodes may complete before others. We track *alive environments* at each timestep. This is critical for RNN policies, as the hidden state is initially sized ``(num_envs x num_agents, hidden_dim)`` but only updated for ``(num_alive_envs x num_agents, hidden_dim)`` when some episodes finish.
- **RNN training** : We use truncated backpropagation through time (TBPTT) to train the RNN network. You can set the length of the sequence using ``tbptt``. 

Logging
-------

We record the following metrics:

- **rollout/ep_reward** : Mean episode reward during environment rollouts.
- **rollout/ep_length** : Mean episode length during rollouts.
- **rollout/num_episodes** : Total number of completed episodes until the current step.
- **rollout/battle_won** (SMAClite only): Fraction of battle won by SMAC agents
- **train/critic_loss** : The critic loss at the current optimization step.
- **train/actor_loss** : The actor loss at the current optimization step.
- **train/actor_gradients** : Magnitude of gradients of actor network.
- **train/critic_gradients** : Magnitude of gradients of critic network.
- **train/num_updates** : Total number of network updates until the current step.
- **eval/ep_reward** : Mean episode reward during evaluation.
- **eval/std_ep_reward** : Standard deviation of episode rewards during evaluation.
- **eval/ep_length** : Mean episode length during evaluation.
- **eval/battle_won** ( SMAClite only): Fraction of battles won during evaluation episodes.

Documentation
-------------

.. py:class:: cleanmarl.maddpg.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, gamma=0.99, buffer_size=5000, batch_size=10, normalize_reward=False, actor_hidden_dim=32, actor_num_layers=1, critic_hidden_dim=128, critic_num_layers=1, train_freq=1, optimizer="Adam", learning_rate_actor=0.0003, learning_rate_critic=0.0003, total_timesteps=1000000, target_network_update_freq=1, polyak=0.005, clip_gradients=-1, log_every=10, eval_steps=50, num_eval_ep=5, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)

    :param env_type: Type of the environment: ``smaclite``, ``pz`` for PettingZoo, etc.
    :type env_type: str

    :param env_name: Name of the environment (``3m``, ``simple_spread_v3``, etc.)
    :type env_name: str

    :param env_family: Environment family when using PettingZoo (``sisl``, ``mpe`` ...).
    :type env_family: str

    :param agent_ids: Include agent IDs (one-hot vector) in observations.
    :type agent_ids: bool

    :param gamma: Discount factor for returns.
    :type gamma: float

    :param buffer_size: Number of episodes in the replay buffer.
    :type buffer_size: int

    :param batch_size: Batch size for training.
    :type batch_size: int

    :param normalize_reward: Normalize the rewards if True.
    :type normalize_reward: bool

    :param actor_hidden_dim: Hidden dimension of the actor network.
    :type actor_hidden_dim: int

    :param actor_num_layers: Number of hidden layers in the actor network.
    :type actor_num_layers: int

    :param critic_hidden_dim: Hidden dimension of the critic network.
    :type critic_hidden_dim: int

    :param critic_num_layers: Number of hidden layers in the critic network.
    :type critic_num_layers: int

    :param train_freq: Train the network each ``train_freq`` episodes of the environment.
    :type train_freq: int

    :param optimizer: Optimizer for both actor and critic.
    :type optimizer: str

    :param learning_rate_actor: Learning rate for the actor network.
    :type learning_rate_actor: float

    :param learning_rate_critic: Learning rate for the critic network.
    :type learning_rate_critic: float

    :param total_timesteps: Total number of environment steps during training.
    :type total_timesteps: int

    :param target_network_update_freq: Update the target network each ``target_network_update_freq`` episode
    :type target_network_update_freq: int

    :param polyak: Polyak coefficient for target network updates.
    :type polyak: float

    :param clip_gradients: ``0<`` for no clipping and ``0>`` to clip gradients at this value.
    :type clip_gradients: float

    :param log_every: Log rollout statistics every ``log_every`` episode.
    :type log_every: int

    :param eval_steps: Evaluate the policy every ``eval_steps`` episode.
    :type eval_steps: int

    :param num_eval_ep: Number of evaluation episodes.
    :type num_eval_ep: int

    :param use_wnb: Enable logging to Weights & Biases if True.
    :type use_wnb: bool

    :param wnb_project: Weights & Biases project name.
    :type wnb_project: str

    :param wnb_entity: Weights & Biases entity name.
    :type wnb_entity: str

    :param device: Device to use (``cpu``, ``gpu``, ``mps``).
    :type device: str

    :param seed: Random seed for reproducibility.
    :type seed: int


.. py:class:: cleanmarl.maddpg_multienvs.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, num_envs=4, gamma=0.99, buffer_size=5000, batch_size=10, normalize_reward=False, actor_hidden_dim=32, actor_num_layers=1, critic_hidden_dim=128, critic_num_layers=1, epochs=4, optimizer="Adam", learning_rate_actor=0.0003, learning_rate_critic=0.0003, total_timesteps=1000000, target_network_update_freq=1, polyak=0.01, clip_gradients=-1, log_every=10, eval_steps=50, num_eval_ep=5, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)

    :param num_envs: Number of parallel environments
    :type num_envs: int

    :param epochs: Number of batches sampled in one update
    :type n_epochs: int

.. py:class:: cleanmarl.maddpg_lstm.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, gamma=0.99, buffer_size=5000, batch_size=10, normalize_reward=False, actor_hidden_dim=32, actor_num_layers=1, critic_hidden_dim=128, critic_num_layers=1, train_freq=1, optimizer="Adam", learning_rate_actor=0.0006, learning_rate_critic=0.0006, total_timesteps=1000000, target_network_update_freq=1, polyak=0.005, clip_gradients=-1, tbptt=10, log_every=10, eval_steps=50, num_eval_ep=5, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)

    :param tbptt: Chunk size for Truncated Backpropagation Through Time (TBPTT).
    :type tbptt: int

.. py:class:: cleanmarl.maddpg_lstm_multienvs.Args(env_type="smaclite", env_name="3m", env_family="mpe", agent_ids=True, num_envs=4, gamma=0.99, buffer_size=5000, batch_size=10, normalize_reward=False, actor_hidden_dim=32, actor_num_layers=1, critic_hidden_dim=128, critic_num_layers=1, optimizer="Adam", learning_rate_actor=0.0003, learning_rate_critic=0.0003, total_timesteps=1000000, target_network_update_freq=1, polyak=0.01, epochs=4, clip_gradients=-1, tbptt=10, log_every=10, eval_steps=50, num_eval_ep=5, use_wnb=False, wnb_project="", wnb_entity="", device="cpu", seed=1)
