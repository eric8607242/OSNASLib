import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.203, 0.1994, 0.2010]

def get_CIFAR_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion=1):
    train_transform = transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True,
                                         download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False,
                                         download=True, transform=test_transform)
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root=dataset_path, train=True,
                                         download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=dataset_path, train=False,
                                         download=True, transform=test_transform)
    else:
        raise 


    if train_portion != 1:
        train_len = len(train_dataset)
        indices = list(range(train_len))
        random.shuffle(indices)
        split = int(np.floor(train_portion * train_len))
        train_idx, val_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=True)
        val_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=val_sampler,
            pin_memory=True)
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True)
        val_loader = None

    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True)
    return train_loader, val_loader, test_loader
