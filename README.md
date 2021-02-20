# OneShot_NAS_example

**...The project is under construction...**

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

## What is neural architecture search (NAS)?
By now, deep learning networks has brought surprising advances in various research field (e.g., image classification, semantic segmentation, and object detection). However, to boost the performance, the model architecture and the model size become more complex and huge significantly, which make the deep learning models are hard to deploy on the devices with limited hardware resource. But designing good neural networks manually specific to various hardware constraints requires much domain knowledge, rich experience on model training and tuning, and lots of time on trial and errors. Besides, the manually designed architectures are often stuck into the local optimal.
To get the better trade off between the hardware resource and the performance, neural architecture search (NAS) aiming at automatically searching the best architecture under the specific hardware constraints is highly demanded.

### The main components of neural architecture search (NAS)
Generally NAS methods can be categorized according to three dimensions: Search space, Search strategy, and Performance estimation strategy.
#### Search Space
For the search space, it mean that the **all candidate neural architectures can be selected** (kernel sizes, channel size, layer size, and etc.). 
#### Search Strategy
Given a search space, the objective of neural architecture search is to **search the best architecture under the specific hardware constraint**. To achieve the goal, a search strategy is needed to be adopted to search the best architecture from the search space. The common search strategies are RL, random search, evolution algorithm, and differentiable.
#### Performance estimation strategy
With the search strategy, **how do we evaluate the architectures in the search space? In other words, how do we know the performace of each architectures?**. To evaluate each architectures in the search space, the performance estimation strategy has to been adopted. The most intuitive estimation strategy is that we train each architecture few epochs (e.g., 10, 5, or 20) to get the approximated perofmrnace of each architecture.


## What is one-shot neural architecture search?

### Single-path one-shot neural architecture search
### Differentiable one-shot neural architecture search

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

