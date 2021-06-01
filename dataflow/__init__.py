import torch

def build_loader(dataset, shuffle, batch_size, num_workers, sampler=None):
    if sampler is not None:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            sampler=sampler
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            shuffle=shuffle,
        )


from .cifar import get_CIFAR_dataloader
from .imagenet import get_imagenet_dataloader
from .face import get_face_dataloader


def get_dataloader(dataset_name, dataset_path, batch_size, num_workers, train_portion=1):
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        return get_CIFAR_dataloader(dataset_name, dataset_path, batch_size, num_workers, train_portion)
    elif dataset_name == "imagenet":
        return get_imagenet_dataloader(dataset_name, dataset_path, batch_size, num_workers, train_portion)
    else:
        raise 


