import sys

from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

from .lscrossentropy import LabelSmoothingCrossEntropy

def get_criterion(name):
    criterion_class = getattr(sys.modules[__name__], name)
    return criterion_class()

def get_hc_criterion(name):
    criterion_class = getattr(sys.modules[__name__], name)
    return criterion_class()

