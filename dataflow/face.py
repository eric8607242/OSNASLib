import os
import os.path as osp

import numpy as np
from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from . import build_loader

FACE_MEAN = [0.485, 0.456, 0.406]
FACE_STD = [0.229, 0.224, 0.225]

class get_pairface_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion=1):
    train_transform = transforms.Compose([
                        transforms.Resize(input_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(FACE_MEAN, FACE_STD)
                        ])
    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(FACE_MEAN, FACE_STD)
                        ])

    train_dataset = ImageFolder(root=osp.join(dataset_path, "face", "train"),
                                transform=train_transform)
    test_dataset = PairFaceDataset(root=osp.join(dataset_path, "face", "test"),
                                transform=test_transform)
    if train_portion != 1:
        train_len = len(train_dataset)
        indices = list(range(train_len))
        random.shuffle(indices)
        split = int(np.floor(train_portion * train_len))
        train_idx, val_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = build_loader(
            train_dataset,
            True,
            batch_size,
            num_workers,
            sampler=train_sampler)
        val_loader = build_loader(
            train_dataset,
            False,
            batch_size,
            num_workers,
            sampler=val_sampler)
    else:
        train_loader = build_loader(
            train_dataset,
            True,
            batch_size,
            num_workers,
            sampler=None)
        val_loader = None
    test_loader = build_loader(test_dataset, False, batch_size, num_workers, sampler=None)

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
