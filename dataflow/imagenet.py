import os 

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_imagenet_dataloader(dataset_path, input_size, batch_size, num_workers, train_portion=1):
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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size+32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


    train_dataset_path = os.path.join(dataset_path, "train")
    val_dataset_path = os.path.join(dataset_path, "val")
    test_dataset_path = os.path.join(dataset_path, "test")

    train_dataset = datasets.ImageFolder(train_dataset_path, train_transform)
    val_dataset = datasets.ImageFolder(val_dataset_path, test_transform)
    test_dataset = datasets.ImageFolder(test_dataset_path, test_transform)

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True)
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True)
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True)

    return train_loader, val_loader, test_loader
