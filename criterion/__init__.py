import sys

from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss


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


# Import customizing module (Do not delete this line)
from .lscrossentropy import LabelSmoothingCrossEntropy
from .tripletloss import OnlineTripletLoss
from .focalloss import FocalLoss
from .faceceloss import FaceCELoss