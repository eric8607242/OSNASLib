import sys

import torch.nn.CrossEntropyLoss as CrossEntropyLoss
import torch.nn.MSELoss as MSELoss

from .lscrossentropy import LabelSmoothingCrossEntropy

def get_criterion(name):
    criterion_class = getattr(sys.modules[__name__], name)
    return criterion_class()

def get_hc_criterion(name):
    criterion_class = getattr(sys.modules[__name__], name)
    return criterion_class()

