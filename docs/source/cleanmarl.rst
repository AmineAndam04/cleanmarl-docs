CleanMARL
=========


CleanMARL is a collection of single-file implementations of Deep Multi-Agent Reinforcement Learning algorithms. We provide standalone and easy-to-follow implementations of state-of-the-art algorithms. For know, we only provide implementations of online algorithms. For each algorithm, we offer multiple implementation variants to test different patterns commonly found in the literature. 
  
algorithms
----------

Currently, we implement the following algorithms:

    - **VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning
    - **QMIX**: Monotonic Value Function Factorization
    - **COMA**: Counterfactual Multi-Agent
    - **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient
    - **FACMAC**: Factored Multi-Agent Centralised Policy Gradients
    - **IPPO**: Independent Proximal Policy Optimization
    - **MAPPO**: Multi-Agent Proximal Policy Optimization

Implementations
---------------

We mainly focus on implementing four variants for each algorithm:

    - Single environment + MLP networks
    - Multiple environments + MLP networks
    - Single environment + RNN networks
    - Multiple environments + RNN networks

A detailed discussion of these variants, as well as other design choices are discussed in :doc:`design` section.

Environments 
------------

CleanMARL currently supports the following environments: 

- **SMAClite**
- **PettingZoo**
- **Level-Based Foraging**

Other environments can be easily added by creating a new environment class inside the ``cleanmarl/env`` folder, following the design of the ``CommonInterface`` base class.


