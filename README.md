# OneShot-NAS-example

## Introduction

* [What is neural architecture search (NAS)](./doc/nas.md)
* [What is one-shot NAS](./doc/one_shot_nas.md)

This is a example repo for one-shot NAS. We cover the basic implementation for one-shot NAS (e.g., supernet, search strategy, and training strategy).
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
```
python3 main.py -c [CONFIG FILE] --title [EXPERIMENT TITLE]
```

More configuration please refer [configuration](./doc/configuration.md).

## Customize NAS For Your Tasks
In this project, you can design different NAS for your task by modifying or expanding any components easily.
Please refer the documents to follow our customizing guidelines.
* [How to customize the criterion](./doc/customize/criterion.md)
* [How to customize the dataloader](./doc/customize/dataloader.md)
* [How to customize the search space](./doc/customize/search_space.md)
* [How to customize the search strategy](./doc/customize/search_strategy.md)
* [How to customize the training strategy](./doc/customize/training_strategy.md)

## TODO
* [ ] EMA for model evaluate
* [ ] Warmup lr scheduler
* [ ] Search space for Single-path NAS
* [ ] Analyze code (`evaluate.py`)
* [ ] Experiment results



