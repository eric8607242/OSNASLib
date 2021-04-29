# Configuration
## Random Search
* `--search-strategy random_search`
* [Optional Hyperparameter]
    * `--random-iteration` : The random sample number of the neural architectures to search the best architecture. (default:1000)
    * `--directly-search` : Directly searching flag. If the flag is `True`, searching the best architecture without training supernet. (Better with the resume flag to load the supernet pretrained weight.)
    

## Evolution Algoeithm
* `--search-strategy evolution`
* [Optional Hyperparameter]
    * `--generation_num` : The generation num to evolve the best architecture. (default:20)
    * `--population` : The population size in each generation to evolve the best architecture. (default:60)
    * `--parent-num` : The parent size to mutate and crossover the best architecture. (default:10)
    * `--directly-search` : Directly searching flag. If the flag is `True`, searching the best architecture without training supernet. (Better with the resume flag to load the supernet pretrained weight.)

## Differentiable Search
* `--search-strategy differentiable`
* [Optional Hyperparameter]
    * `--sample-strategy` : The strategy to train supernet [uniform, fair, differentiable]. If you adopt uniform or fair, we recommend that set `--bn-track-running-stats` as `0` to close the running stats tracking.
    * `--a-optimizer` : The optimzier for the architecture parameters. (default:sgd)
    * `--a-lr` : The learning rate for the architecture parameters. (default:0.05)
    * `--a-weight-decay` : The weight decay for the architecture parameters. (default:0.0004)
    * `--a-momentum` : The momentum for the architecture parameters. (default:0.9)

## Evaluate - Train from scratch
* `--searched_model_path [PATH TO SEARCHED MODEL]`


## Config
### Common Config
* `--title`
    * `--resume [PATH TO RESUME CHECKPOINT]`
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
* `--criterion-type`

* Path
    * `--root-path` : The root path to save all of checkpoint, lookup-table, or searched model in this searching process. We set `root-path` based on the `title` and `random seed`.
    * `--logger-path` : The path to logger file.
    * `--writer-path` : The path to tensorboard writer file.
    * `--checkpoint-path-root` : The path to the directoy of the checkpoint. We seperate the directory into `evaluate/` and `search/` for different process.
    * `--lookup-table-path` : The path to lookup table.
    * `--best-model-path` : The path to the weight of best model.
    * `--searched-model-path` : The path to the configuration of the searched model.
    * `--hyperparameter-tracker` : The path to the hyperparameter-tracker. Hyperparameter-tracker record all hyperparameter for each searching and evaluating process.



