import os
import os.path as osp
import random

import numpy as np
from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

FACE_MEAN = [0.5, 0.5, 0.5]
FACE_STD = [0.5, 0.5, 0.5]

def get_face_angular_dataloader(dataset_path, input_size, batch_size, num_workers, train_portion=1):
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
                        transforms.Resize(input_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(FACE_MEAN, FACE_STD)
                        ])
    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])

    train_dataset = datasets.ImageFolder(root=osp.join(dataset_path, "face", "train", "CASIA-WebFace"),
                                transform=train_transform)
    test_dataset = PairFaceDataset(root=osp.join(dataset_path, "face", "test", "LFW"),
                                transform=test_transform)
    if train_portion != 1:
        train_len = len(train_dataset)
        indices = list(range(train_len))
        random.shuffle(indices)
        split = int(np.floor(train_portion * train_len))
        train_idx, val_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True)
        val_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=val_sampler,
            pin_memory=True)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True)
        val_loader = None

    test_loader = DataLoader(
            dataset=test_dataset, 
            shuffle=False, 
            batch_size=batch_size, 
            num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

class PairFaceDataset:
    """Pairing Faces for evaluation"""
    def __init__(self, root, transform=None):
        # Save arguments
        self.root = root
        self.transform = transform
        # Read dataset
        self.data = np.load(osp.join(root, 'data.npy'))
        self.label = np.load(osp.join(root, 'label.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        img1 = Image.fromarray(pair[0])
        img2 = Image.fromarray(pair[1])
        label = self.label[idx]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return (img1, img2), label
