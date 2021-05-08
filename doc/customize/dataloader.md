# How to customize the dataloader

## API
```python
def get_your_dataloader(dataset_name, dataset_path, batch_size, num_workers, train_portion=1):

    train_loader = 
    val_loader = 
    test_loader =

    return train_loader, val_loader, test_loader
```
