import sys

from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

from .lscrossentropy import LabelSmoothingCrossEntropy
from .tripletloss import OnlineTripletLoss

def get_criterion(name, criterion_config):
    criterion_class = getattr(sys.modules[__name__], name)
    try:
        return criterion_class(criterion_config=criterion_config)
    except:
        return criterion_class()

def get_hc_criterion(name, criterion_config):
    criterion_class = getattr(sys.modules[__name__], name)
    try:
        return criterion_class(criterion_config=criterion_config)
    except:
        return criterion_class()

