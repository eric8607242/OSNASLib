# OneShot_NAS_example

**...The project is under construction...**

## Introduction

* [What is neural architecture search (NAS)](./doc/nas.md)
* [What is one-shot NAS](./doc/one_shot_nas.md)

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
* [x] search strategy (e.g., evolution algorithm and random search)
* [ ] hardware constraint loss for differentiable search strategy.

### Util function
* [x] Average tracker
* [x] Get logger, get tensorboard writer
* [x] Calculate classification accuracy
* [x] Argument parser

## Config
```
cd config_file/arg_config/
```
* `search_config.py` : Config of search process (e.g., search strategy, epoch, target hc, and epoch)
    * `--search-strategy`: Search strategy for searching the best architecture. (e.g., Random search, evolution algorithm, and differentiable)
    * `--search-space` : Search space in different papaer (e.g., ProxylessNAS, FBNet, and SPOS)
    * `--sample-strategy` : The way to train supernet (e.g., Uniform sampling, Fairstrict sampling,and differentiable)
        * differentiable : Jointly search architecture and training supernet with differentiable sheme.
    * `--hc-weight` : The weight of hardware constraint objective. (default : 0.005)

## Search
### Random Search
```
python3 search.py --title [EXPERIMENT TITLE] --search-strategy random_search
```
* [Optional Hyperparameter]
    * `--random-iteration` : The random sample number of the neural architectures to search the best architecture. (default:1000)

### Evolution Algoeithm
```
python3 search.py --title [EXPERIMENT TITLE] --search-strategy evolution
```
* [Optional Hyperparameter]
    * `--generation_num` : The generation num to evolve the best architecture. (default:20)
    * `--population` : The population size in each generation to evolve the best architecture. (default:60)
    * `--parent-num` : The parent size to mutate and crossover the best architecture. (default:10)

### Differentiable Search
```
python3 search.py --title [EXPERIMENT TITLE] --search-strategy differentiable
```
* [Optional Hyperparameter]
    * `--a-optimizer` : The optimzier for the architecture parameters. (default:sgd)
    * `--a-lr` : The learning rate for the architecture parameters. (default:0.05)
    * `--a-weight-decay` : The weight decay for the architecture parameters. (default:0.0004)
    * `--a-momentum` : The momentum for the architecture parameters. (default:0.9)

