# OSNASLib

## Introduction
OSNASLib is a library for one-shot NAS. Recently, searching the neural architecture for various deep learning tasks becomes a common pipeline no matter in semantic segmentation, object detection, NLP, etc. Therefore, OSNASLib provides extremely flexible interfaces to allow researchers can incorporate different one-shot NAS algorithms into various specific tasks efficient.

In OSNASLib, we cover various components of one-shot NAS (e.g., search space, search strategy, and training strategy). You can incorporate any components to specify for different dataset, differnt criterion and different specific tasks easily. 
Please refer to [What is OSNASLib](./doc/osnaslib.md) for more detail about this library.

* [What is neural architecture search (NAS)](./doc/nas.md)
* [What is one-shot NAS](./doc/one_shot_nas.md)

**We are glad at all contributions to improve this repo. Please feel free to pull request.**

## Getting Started
```
python3 main.py -c [CONFIG FILE] --title [EXPERIMENT TITLE]
```
* `[CONFIG FILE]` : The path to the config file. We provide serveral example config files in `./config/arg_config/`.
* `[EXPERIMENT TITLE]` : In each experiment, all corresponding files will be saved in the directory named with experiment title. 

More information about configuration please refer to [configuration](./doc/configuration.md).

## Customize NAS For Your Tasks
OSNASLib provides extremely flexible interfaces to make researchers can incorporate different components of one-shot NAS for various tasks easily.
Please refer to the documents to follow our customizing guidelines.
* [How to customize the criterion](./doc/customize/criterion.md)
* [How to customize the dataloader](./doc/customize/dataloader.md)
* [How to customize the search space](./doc/customize/search_space.md)
* [How to customize the search strategy](./doc/customize/search_strategy.md)
* [How to customize the training strategy](./doc/customize/training_strategy.md)

## TODO
* [ ] Reconstruct criterion
* [ ] Reconstruct model (e.g., model constructor and model config file)
* [ ] Documents

