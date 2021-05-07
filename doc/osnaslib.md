# What we cover in this library

In OSNASLib, we cover the basic implementation for one-shot NAS (e.g., supernet, search strategy, and training strategy).

* Components implemented in this repo
    * Supernet
        * Search space of ProxylessNAS
        * Search space of FbNet
        * Search space of SPOS
    * Supernet training strategy
        * Uniform sampling
        * Strict fairness
        * Differentiable (specific for differentiable search strategy)
    * Search Strategy
        * Gradient descent(Differentiable)
            - Softmax
            - Gumbel Softmax
        * Random search
        * Evolution algorithm
        * Architecture generator
* Additional property in OSNASLib
    * Hyperparameter tracker
        * You can record all hyperparameter in each searching or evaluating process.
    * Evaluate searched neural architecture. (Train from scratch)
