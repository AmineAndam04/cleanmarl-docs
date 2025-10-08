CleanMARL
=========


CleanMARL is a collection of single-file implementations of Deep Multi-Agent Reinforcement Learning algorithms. We provide standalone and easy-to-follow implementations of state-of-the-art algorithms. For know, we only provide implementations of online algorithms in cooperative settings. For each algorithm, we provide multiple implementations variants to test different implementation patterns found in the literature. We provide the implementation of the following algorithms:

    - VDN: Value-Decomposition Networks For Cooperative Multi-Agent Learning
    - QMIX: Monotonic Value Function Factorization
    - COMA: Counterfactual Multi-Agent
    - MADDPG: Multi-Agent Deep Deterministic Policy Gradient
    - FACMAC: Factored Multi-Agent Centralised Policy Gradients
    - IPPO: Independent Proximal Policy Optimization
    - MAPPO: Multi-Agent Proximal Policy Optimization

We mainly focus on implementing 4 variants for each algorithm:

    - Single environment + MLP networks
    - Multiple environments + MLP networks
    - Single environment + RNN networks
    - Multiple environments + RNN networks

A detailed discussion of these variants, as well as other design choices are discussed in :doc:`design` section.
