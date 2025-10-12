Training details
================

Replay buffers
--------------

In single-agent RL, the replay buffer always stores environment transitions in the form ``(obs, action, reward, done, next_obs)``.  
However, many MARL algorithms use replay buffers that store **entire episodes** instead.  
Except for VDN, we use episode-based replay buffers for the implemented off-policy algorithms (QMIX, MADDPG, FACMAC).

Episode-based replay buffers are easier to handle when training **recurrent policies**, since we have access to full sequences and can use accurate hidden states.  
In transition-based buffers, as we did for VDN, we store sequences of transitions.  
When we sample a sequence to update the policy, we first initialize the hidden state ``h = None``.  
Then, we use the first ``burn_in`` steps to warm up the hidden state (without backpropagation), and use the rest of the sequence for the policy update.

.. code-block:: python

    h_target = None
    h_utility = None
    with torch.no_grad(): ## First burn some steps 
        for t in range(args.burn_in):
            ...
            _,h_target =  target_network(batch_next_obs_t,h = h_target)
            _,h_utility = utility_network(batch_obs_t,h=h_utility)
    loss = 0 ## We can start backpropagating
    for t in range(args.burn_in, args.seq_length):
        with torch.no_grad():
            ...
            q_next,h_target = target_network(batch_next_obs_t,h=h_target,avail_action =batch_next_avail_action_t )
            ...
            targets = batch_reward[:,t] + args.gamma * (1-batch_done[:,t])*vdn_q_max

        ...
        q_values,h_utility = utility_network(batch_obs_t,h=h_utility)
        ...
        loss += F.mse_loss(targets,vqn_q_values)
    loss = loss / (args.seq_length - args.burn_in)
    optimizer.zero_grad()
    loss.backward()


Parallel environments
---------------------

Parallel environments are particularly useful for **on-policy** algorithms, since we need to collect new trajectories at each training step using the current policy.  
For **off-policy** algorithms, however, the training data is sampled from a replay buffer, so how fast we fill the buffer matters less because off-policy methods can be trained using data from older policies.

Parallel environments may even perform worse compared to single-environment versions.  
If we update the policies every ``train_freq`` steps in a single environment setting, we would effectively update them every ``num_parallel_envs * train_freq`` steps in a parallel setup.  
To compensate for this, instead of backpropagating once per training step, we perform **multiple epochs of training** per step in the parallel setting.  
We also keep track of ``num_updates`` to monitor the total number of policy updates for each algorithm.

When using multiple environments in parallel with recurrent policies, some episodes may complete before others. We track *alive environments* at each timestep. This is critical for RNN policies, as the hidden state has an initial size of ``(num_envs x num_agents, hidden_dim)`` but eventually becomes of shape ``(num_alive_envs x num_agents, hidden_dim)`` when some episodes finish.

TD returns & advantages
-----------------------

For TD(λ) returns use the recursive formula from `Reconciling λ-Returns with Experience Replay (Equation 3) <https://arxiv.org/pdf/1810.09967>`_ . We start by :math:`R^{\lambda}_T = 0`

.. math::

   \begin{align}
   R^{\lambda}_t &= R^{(1)}_t + \gamma \lambda \Big[ R^{\lambda}_{t+1} - \max_{a' \in \mathcal{A}} Q(\hat{s}_{t+1}, a') \Big] \\
   &= r_t + \gamma  \Big[ \lambda R^{\lambda}_{t+1} + (1-\lambda) \max_{a' \in \mathcal{A}} Q(\hat{s}_{t+1}, a') \Big]
   \end{align}

We don’t handle time-outs except in the ``coma_lbf.py`` implementation, where we keep track of whether the last step is *truncated* or *done*.  
If it’s truncated, we add the action-value of the last observation.


We don't directly estimate the advantages using GAE estimates, we instead use the TD(λ) returns by exploiting the following formula that can be found in  `page 47 in David Silver's lecture n 4 <https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-4-model-free-prediction-.pdf>`_ 

.. math::

  A(s_t,a_t) = R^{\lambda}_t -V(s_t)


Recurrent policies
------------------

Instead of unrolling recurrent policies over the entire episode, we split them into **short chunks** and backpropagate only over those chunks.  
The chunk size is configured using the ``tbptt`` argument.  
After ``tbptt`` steps, the hidden state is detached.  
To unroll over the full episode, set ``tbptt`` to the maximum episode length (the environment’s ``time_limit``).


.. code-block:: python

            truncated_actor_loss = None
            actor_loss_denominator = None
            T = None
            for t in range(b_obs.size(1)):
                ...
                actor_loss = -pg_loss - args.entropy_coef*entropy_loss
                total_actor_loss += actor_loss
                if truncated_actor_loss is  None:
                    truncated_actor_loss = actor_loss
                    actor_loss_denominator = (b_mask[:,t].sum())
                    T = 1
                else:
                    truncated_actor_loss += actor_loss
                    actor_loss_denominator += (b_mask[:,t].sum())
                    T += 1
                if ((t+1) % args.tbptt == 0) or (t == (b_obs.size(1)-1)):
                    truncated_actor_loss = truncated_actor_loss/(actor_loss_denominator*T)
                    actor_optimizer.zero_grad()
                    truncated_actor_loss.backward()
                    h = h.detach()
