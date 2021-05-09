# How to customize the dataloader
In OSNASLib, we cover serveral basic dataloader (e.g., CIFAR10, CIFAR100, and ImageNet). Besides, we also provide flexible API interface to customize for any your specific tasks or the novel dataloder. In this document, we will briefly introduce how to customize the dataloader in `./dataflow/` easily.

## Create `customize.py` File
First of all, you should create a new file for your dataloader.
```
touch ./dataflow/customize.py
```

## Dataloader Function
To customize for your dataloader, you should create a API function which returns training dataloader, validation loader and testing loader. We provide a example to create the function as following.

```python
# ./dataflow/customize.py
from . import build_loader

def get_your_dataloader(dataset_name, dataset_path, batch_size, num_workers, train_portion=1):

    train_loader = 
    val_loader = 
    test_loader =

    return train_loader, val_loader, test_loader
```

## Import Your Dataloader Function
After creating your training strategy, the class should be imported to in `./training_strategy/__init__.py`.

```python3
# ./dataflow/__init__.py
from .customize import get_your_dataloader
```

After importing the customizing function, you should add the option in `get_loader` for your dataloader.

```python3
# ./dataflow/__init__.py

def get_dataloader(dataset_name, dataset_path, batch_size, num_workers, train_portion=1):
    ...
    elif dataset_name == "customize":
        return get_your_dataloader(dataset_name, dataset_path, batch_size, num_workers, train_portion)
```


## Setting The Dataloder In Config File
To utilize the customizing dataloader, you should set the agent in the config file as following.

```python3
dataset:
    dataset: "customize"
```
