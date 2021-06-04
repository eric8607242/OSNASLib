import torch

from .cifar import get_cifar100_dataloader, get_cifar10_dataloader
from .imagenet import get_imagenet_dataloader
from .face import get_face_dataloader


def get_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion=1):
    dataloader_builder = getattr(sys.modules[__name__], dataset_name)

    return dataloader_builder(dataset_path, input_size, batch_size, num_workers, train_portion)


