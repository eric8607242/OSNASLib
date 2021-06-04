import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

def get_{{customize_name}}_dataloader(dataset_path, input_size, batch_size, num_workers, train_portion=1):
    # Write your code here
    return train_loader, val_loader, test_loader
