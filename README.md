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
* Supernet training strategy
    * Uniform sampling
    * Strict fairness
* Search Strategy
    * Gradient Descent(Differentiable)
        - Softmax
        - Gumbel Softmax
    * Random search
    * Evolution algorithm


We are glad at all contributions to improve this repo. Please feel free to pull request.

## TODO
* [ ] Hyperparameter record for each run
* [ ] EMA for model evaluate
* [ ] Warmup lr scheduler
* [ ] Search space for Single-path NAS
* [ ] Evaluate supernet (`evaluate.py`)
* [ ] Experiment results

## Config
### Common Config
Following is the common config in `config_file/arg_config/search_config.py` and `config_file/arg_config/evaluate_config.py`. 
* `--title`
    * `--resume`
    * `--random-seed`
    * `--device`
    * `--ngpu`
* `--optimizer`
    * `--lr`
    * `--weight-decay`
    * `--momentum`
* `--lr-scheduler`
    * `--decay-step`
    * `--decay-ratio`
    * `--alpha`
    * `--beta`
* `--dataset`
    * `--dataset-path`
    * `--classes`
    * `--input-size`
    * `--batch-size`
    * `--num-workers`
    * `--train-portion`


### Config File
```
cd config_file/arg_config/
```
* `search_config.py` : Config for search process (e.g., search strategy, epoch, target hc, and epoch)
    * `--search-strategy`: Search strategy for searching the best architecture. (e.g., Random search, evolution algorithm, and differentiable)
    * `--search-space` : Search space in different papaer (e.g., ProxylessNAS, FBNet, and SPOS)
    * `--sample-strategy` : The way to train supernet (e.g., Uniform sampling, Fairstrict sampling,and differentiable)
        * differentiable : Jointly search architecture and training supernet with differentiable sheme.
    * `--hc-weight` : The weight of hardware constraint objective. (default : 0.005)
    * `--bn-momentum`
    * `--bn-track-running-stats` : Whether tracking the running stats for BN or utilizing the stats of batch data in each iteraion. (0 : False, 1 : True)

* `evaluate_config.py` : Config for evaluate searched architecture.
    *

## Search
### Random Search
```
python3 search.py --title [EXPERIMENT TITLE] --search-strategy random_search
```
* [Optional Hyperparameter]
    * `--random-iteration` : The random sample number of the neural architectures to search the best architecture. (default:1000)
    * `--directly-search` : Directly searching flag. If the flag is `True`, searching the best architecture without training supernet. (Better with the resume flag to load the supernet pretrained weight.)
    

### Evolution Algoeithm
```
python3 search.py --title [EXPERIMENT TITLE] --search-strategy evolution
```
* [Optional Hyperparameter]
    * `--generation_num` : The generation num to evolve the best architecture. (default:20)
    * `--population` : The population size in each generation to evolve the best architecture. (default:60)
    * `--parent-num` : The parent size to mutate and crossover the best architecture. (default:10)
    * `--directly-search` : Directly searching flag. If the flag is `True`, searching the best architecture without training supernet. (Better with the resume flag to load the supernet pretrained weight.)

### Differentiable Search
```
python3 search.py --title [EXPERIMENT TITLE] --search-strategy differentiable
```
* [Optional Hyperparameter]
    * `--sample-strategy` : The strategy to train supernet [uniform, fair, differentiable]. If you adopt uniform or fair, we recommend that set `--bn-track-running-stats` as `0` to close the running stats tracking.
    * `--a-optimizer` : The optimzier for the architecture parameters. (default:sgd)
    * `--a-lr` : The learning rate for the architecture parameters. (default:0.05)
    * `--a-weight-decay` : The weight decay for the architecture parameters. (default:0.0004)
    * `--a-momentum` : The momentum for the architecture parameters. (default:0.9)

## Evaluate
Train the searched architectrue from scratch to evaluate the search quality.
```
python3 train.py --title [EXPERIMENT TITLE] --searched_model_path [PATH TO SEARCHED MODEL]
```

