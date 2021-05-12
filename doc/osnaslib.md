# Overview of OSNASLib
In OSNASLib, we totally have 7 parts. And users can modify each of them to evolve to different tasks easily without requiring codebase reconstruct. 

1. Search Strategy
2. Dataflow
3. Training Strategy
4. Criterion
5. Model
6. Agent
7. Config

In the following sections, we will introduce the code structure of each part clearly.

## Search Strategy
Search strategy is one of the most important components in NAS. Given a search space, the search strategy has to explore the search space and search the best architecture from the search space. At default, we cover the most common search strategy as follows.

* Gradient descent(Differentiable)
    - Softmax
    - Gumbel Softmax
* Random search
* Evolution algorithm
* Architecture generator

In OSNASLib, all search straties have to inherit the abstract class `BaseSearcher` to implement the abstract method. With the implementation of abstract method, users can incorporate any search strategy into entire searching pipeline easily.


## Dataflow

## Training Strategy
* Uniform sampling
* Strict fairness
* Differentiable (specific for differentiable search strategy)

## Criterion

## Model
* Search space of ProxylessNAS
* Search space of FbNet
* Search space of SPOS

## Agent

## Config


* Additional property in OSNASLib
    * Hyperparameter tracker
        * You can record all hyperparameter in each searching or evaluating process.
    * Evaluate searched neural architecture. (Train from scratch)
