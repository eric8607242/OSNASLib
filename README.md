# OneShot-NAS-example

## Introduction

* [What is neural architecture search (NAS)](./doc/nas.md)
* [What is one-shot NAS](./doc/one_shot_nas.md)

The is a example repo for one-shot NAS. We cover the basic implementation for one-shot NAS(e.g., supernet, search strategy, and training strategy).
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
        * Gradient Descent(Differentiable)
            - Softmax
            - Gumbel Softmax
        * Random search
        * Evolution algorithm
* Additional property in this repo
    * Hyperparameter tracker
        * We record all hyperparameter in each searching or evaluating process.
    * Evaluate searched neural architecture. (Train from scratch)

We are glad at all contributions to improve this repo. Please feel free to pull request.

## Getting Started
```python
# Search the best architecture under the specific hardware constraint.
searcher = Searcher(args)
searcher.search()

# Evaluate the searched architecture.
evaluator = Evaluator(args)
evaluator.evaluate()
```

More configuration please refer [configuration](./doc/configuration.md).

## TODO
* [ ] EMA for model evaluate
* [ ] Warmup lr scheduler
* [ ] Search space for Single-path NAS
* [ ] Analyze code (`evaluate.py`)
* [ ] Experiment results



