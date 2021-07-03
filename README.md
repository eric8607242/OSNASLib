# OSNASLib

## Introduction
OSNASLib is a general one-shot NAS framework empowering uses to incorporate one-shot NAS methods into various tasks (e.g. face recongition) easily. OSNASLib consists of six major components: criterion, dataflow, training agent, search space, search strategy, and training strategy.
The framework of OSNASLib is illustrated as following:
![osnaslib](./resource/osnaslib_abstract.png)
> The regions of blue lines are the components that user can customize easily with our interface.

For each component, OSNASLib provides serveral baselines and allows users to customize them for various tasks flexibly. To build a deep neural network with one-shot NAS, users only need to set the training agent, dataflow, and criterion without implementing functional details of one-shot NAS. Furthermore, users can develop new one-shot NAS methods with OSNASLib by customizing search space, search strategy, and training strategy. With the provided baselines, OSNASLib allows users to benchmark their proposed NAS methods fairly (e.g., same codebases and same evaluation configurations).

* [What is neural architecture search (NAS)](./doc/nas.md)
* [What is one-shot NAS](./doc/one_shot_nas.md)
* [The major components in OSNASLib](./doc/osnaslib.md)

### Who Needs OSNASLib?
* NAS beginners who want to build the codebases of basic one-shot NAS methods quickly.
* Researchers who want to integrate NAS methods into a targeted task to improve performance.
* Researchers who focus on desinging NAS methods for various tasks (e.g., image classification and face recognition) and compare with other baseline methods fairly.

**We are glad at all contributions to improve this repo. Please feel free to pull request.**

## Requirements
* Python >= 3.6
* torch >= 1.5.0

Please clone the repo and install the corresponding dependency.
```
git clone https://github.com/eric8607242/OSNASLib
pip install -r requirements.txt
```

## Getting Started
```
python3 main.py -c [CONFIG FILE] --title [EXPERIMENT TITLE]

optional arguments:
    --title                 The title of the experiment. All corrsponding files will be saved in the directory named with experiment title.
    -c, --config            The path to the config file. Refer to ./config/ for serveral example config file.
```
### Classification
``` python3 
python3 main.py -c ./config/classification/uniform_evolution.yml --title uniform_sampling_evolution_search
```

### Face Recognition
Before searching architecture for face recognition, please download the dataset first.
```bash
bash ./script/facedata_download.sh
```
> Thanks to Johnnylord for the support of face recognition training pipeline.
```bash
python3 main.py -c ./config/face_recognition/uniform_evolution.yml --title uniform_sampling_evolution_search
```

OSNASLib provides several example configurations for utilizing different baseline components. Please refer to `./config/` for more information about configuration.

## Customize NAS For Your Tasks
### Generate Templare
OSNASLib empowers users to customize specific components for various tasks. However, importing and cooperating multiple components into the main agent are still tedious. To reduce this burden, OSNASLib provides the interface generator to generate interfaces for each component and automatically import to the main agent. To generate interfaces, only one command is needed. 
```
python3 build_interface.py -it [INTERFACE TYPE] --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]

optional arguments:
    -it, --interface-type   The type of the generated interface (e.g., agent, criterion, and dataflow).
    --customize-name        The filename of the customizing interface.
    --customize-class       The classname of the interface class in customizing interface.
```

Please refer to the documents for more detail of customizing.
* [How to customize the criterion](./doc/customize/criterion.md)
* [How to customize the dataloader](./doc/customize/dataloader.md)
* [How to customize the search space](./doc/customize/search_space.md)
* [How to customize the search strategy](./doc/customize/search_strategy.md)
* [How to customize the training strategy](./doc/customize/training_strategy.md)
* [How to customize the agent](./doc/customize/agent.md)

In OSNASLib, we provide the example for serveral tasks. Please reference for following documents for more detail about the example:
* [Classification](./doc/example/classification.md)
* [Face Recognition](./doc/example/face_recognition.md)

## Related Resources
* [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects)
* [Awesome-AutoDL](https://github.com/D-X-Y/Awesome-AutoDL)
* [AutoML.org](https://www.automl.org/)


