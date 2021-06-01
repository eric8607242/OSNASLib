import torch

from .cifar import get_CIFAR_dataloader
from .imagenet import get_imagenet_dataloader
from .face import get_face_dataloader


def get_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion=1):
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        return get_CIFAR_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion)
    elif dataset_name == "imagenet":
        return get_imagenet_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion)
    elif dataset_name == "face":
        return get_face_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion)
    else:
        raise 


