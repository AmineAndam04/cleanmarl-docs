Multi-Agent Proximal Policy Optimization
========================================


    - Paper link:  `IPPO <https://arxiv.org/abs/2103.01955>`_ 

Quick facts:
    - MAPPO trains a centralized critic and decentralized actors.



Ihe unique difference between MAPPO and IPPO is that MAPPO uses a centralizes critic :math:`V(s;\phi)`, instead of a decentralized critic :math:`V_i(o_i;\phi)`



.. image:: ../_static/mappo_network.png
   :alt: Architecture diagram
   :width: 500px
   :align: center

Pseudocode
----------
.. image:: ../_static/mappo_algorithm.svg
   :alt: Architecture diagram
   :width: 100%
   :align: center
