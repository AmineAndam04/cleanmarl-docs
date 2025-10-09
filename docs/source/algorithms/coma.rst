Counterfactual Multi-Agent
==========================

    - Paper link:  `COMA <https://arxiv.org/abs/1705.08926>`_ 

Quick facts:
    - COMA is an on-policy actor-critic algorithm. 
    - COMA uses a centralized critic with a decentralized actors.

Key ideas:


A straightforward adaptation of single actor-critic algorithms to multi-agent RL would be to have for each agent an actor :math:`\pi_i(a_i| o_i; \theta)` and a critic :math:`Q_i(o_i,a_i;\phi)` using the following losses:

.. math::

    \mathcal{L}^{actor}_i(\theta) = - A(o_i,a_i;\phi) log(\pi_i(a_i| o_i; \theta)) 

.. math::

    \mathcal{L}^{critic}_i(\phi) = (y - Q_i(o_i,a_i;\phi))^2


The two mean ideas of COMA is 

 1. To use only one centralized critic  :math:`Q(\mathbf{s}, \mathbf{o},\mathbf{a})`
 2. Replace the standard advantage :math:`A(o_i,a_i)` with an individual *counterfactual advantage* that compares the action-value of an agent' action :math:`a_i` to the expect action-value if the agent had selects other actions, while others' actions are fixed.

Concretely, we use the compute following advantage for each agent :math:`i \in \mathcal{I}` :

.. math::
        A_i(\mathbf{s}, \mathbf{o},\mathbf{a}) = Q(\mathbf{s}, \mathbf{o},\mathbf{a}) - \sum_{a'_i} \pi_i(a'_i|o_i) Q(\mathbf{s}, \mathbf{o},(\mathbf{a}_{-i},a'_i))

implementation-wise, the centralized Q-network takes as input :math:`\mathcal{s}`, the agent's observation :math:`o_i` and the other :math:`n-1` agents actions :math:`\mathbf{a}_{-i}` . The output the of network has the same size as the action-space, we output a value for each possible action :math:`a_i`. This architectures allows us to compute the counterfactual advantage easily.


.. image:: ../_static/coma_network.png
   :alt: Architecture diagram
   :width: 700px
   :align: center


Pseudocode
----------

.. image:: ../_static/coma_algorithm.svg
   :alt: Architecture diagram
   :width: 100%
   :align: center