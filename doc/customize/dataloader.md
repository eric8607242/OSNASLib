# How To Customize Dataflow
In OSNASLib, we cover serveral basic dataloader (e.g., CIFAR10, CIFAR100, and ImageNet). Besides, we also provide flexible API interface to customize for any your specific tasks or the novel dataloder. In this document, we will briefly introduce how to customize the dataloader in `./dataflow/` easily.

## Generate Interface
```
python3 build_interface.py -t dataflow --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the criterion interface, the file `CUSTOMIZE NAME` will be created in `criterion/`, and the corresponding import will be create automatically.

### Interface Struture
```
- dataflow/
    |- [CUSTOMIZE NAME].py
        ...
```

## Dataflow Interface
For customizing dataflow, the interface should return three pytorch dataloders which are `train_loader`, `val_loader` and `test loader`.

```python3
# ./dataflow/[CUSTOMIZE NAME].py
def get_{{customize_name}}_dataloader(dataset_path, input_size, batch_size, num_workers, train_portion=1):
    """ Prepare dataset for training and evaluating pipeline

    Args:
        dataset_path (str)
        input_size (int)
        batch_size (int)
        num_workers (int)
        train_portion (float)

    Return:
        train_loader (torch.utils.data.DataLoader)
        val_loader (torch.utils.data.DataLoader)
        test_loader (torch.utils.data.DataLoader)
    """
    # Write your code here
    return train_loader, val_loader, test_loader
```

## Setting Config File
After customizing for your dataflow, you can utilize your dataflow by setting the dataset in the config file easily.

```python3
dataset:
    dataset: "[CUSTOMIZE_NAME]"
```

