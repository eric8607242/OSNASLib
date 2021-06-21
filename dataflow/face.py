import os
import os.path as osp
import random

import numpy as np
from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

FACE_MEAN = [0.5, 0.5, 0.5]
FACE_STD = [0.5, 0.5, 0.5]

def get_face_dataloader(dataset_path, input_size, batch_size, num_workers, train_portion=1):
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
                        transforms.Normalize(FACE_MEAN, FACE_STD)
                        ])

    train_dataset = datasets.ImageFolder(root=osp.join(dataset_path, "face", "train", "CASIA-WebFace"),
                                transform=train_transform)
    test_dataset = PairFaceDataset(root=osp.join(dataset_path, "face", "test", "LFW"),
                                transform=test_transform)
    if train_portion != 1:
        train_len = len(train_dataset)
        labels = np.array([s[1] for s in train_dataset.samples])

        indices = np.arange(train_len)
        random.shuffle(indices)
        split = int(np.floor(train_portion * train_len))

        train_idx, val_idx = indices[:split], indices[split:]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_sampler = BalancedBatchSampler(train_idx, train_labels, batch_size)
        val_sampler = BalancedBatchSampler(val_idx, val_labels, batch_size)

        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_sampler=train_sampler)
        val_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_sampler=val_sampler)
    else:
        train_len = len(train_dataset)
        labels = np.array([s[1] for s in train_dataset.samples])

        train_idx = np.arange(train_len)
        random.shuffle(train_idx)
        
        train_labels = labels[train_idx]
        
        train_sampler = BalancedBatchSampler(train_idx, train_labels, batch_size)

        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_sampler=train_sampler)
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


class BalancedBatchSampler(Sampler):
    def __init__(self, data_idx, labels, batch_size, P=32):
        self.labels = labels
        self.n_samples = len(labels)
        self.P = P
        self.K = batch_size // P
        self.batch_size = batch_size

        self.data_idx = data_idx

        # Construct lookup table
        self.label_set = list(set(labels))
        self.label_to_indices = {label: data_idx[np.where(np.array(labels) == label)[0]].tolist()
                                    for label in self.label_set}
        
        for l in self.label_set:
            np.random.shuffle(self.label_to_indices[l])

        # dynamic information
        self.used_label_indices_count = { label: 0 for label in self.label_set }
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count + self.batch_size > self.n_samples:
            raise StopIteration

        target_labels = np.random.choice(self.label_set, self.P, replace=False)
        indices = []
        for target_label in target_labels:
            search_ptr = self.used_label_indices_count[target_label]
            indices.extend(self.label_to_indices[target_label][search_ptr:search_ptr+self.K])
            self.used_label_indices_count[target_label] += self.K

            if self.used_label_indices_count[target_label] + self.K > len(self.label_to_indices[target_label]):
                np.random.shuffle(self.label_to_indices[target_label])
                self.used_label_indices_count[target_label] = 0

        self.count += self.batch_size
        return indices

    def __len__(self):
        return self.n_samples // self.batch_size
