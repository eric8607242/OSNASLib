# OneShot_NAS_example
## Introduction
The is a example repo for one-shot NAS. We cover the basic implementation for one-shot NAS(e.g., supernet, search strategy, and training strategy).

**The main components of one-shot NAS in this repo**
* Supernet
    * Search space of ProxylessNAS
    * Search space of FbNet
    * Search space of SPOS
    * Search space of Single-path NAS
* Supernet training strategy
    * Uniform sampling
    * Strict fairness
* Search Strategy
    * Gradient Descent(Differentiable)
        - Softmax
        - Gumbel Softmax
    * Random search
    * Evolution algorithm

## TODO
### Search Space
* [x] supernet architecture for FBNet
* [x] supernet architecture for ProxylessNAS
* [x] supernet architecture for SPOS
* [x] supernet forward process(e.g., differentiable and single-path)
    - gumbel softmax
    - softmax
    - specific architecture(for uniform sampling and fairness sampling)
* [x] contruct lookup table 
    - FLOPs 
    - Parameter number

### Search strategy
* [x] dataflow(e.g., CIFAR10, CIFAR100, and ImageNet)
* [x] sample strategy(e.g., uniform sample, strictfairness sample)
* [x] get optimizer(e.g., rmsprop, sgd, and adam)
* [x] get scheduler(e.g., cosine and step scheduler)

### Util function
* [x] Average tracker
* [x] Get logger, get tensorboard writer
* [x] Calculate classification accuracy
* [x] Argument parser
