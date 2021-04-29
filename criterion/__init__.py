import torch.nn as nn

from .loss import cross_entropy_with_label_smoothing

def get_criterion(criterion_type):
    if criterion_type == "CESmooth":
        return cross_entropy_with_label_smoothing
    elif criterion_type == "CE":
        return nn.CrossEntropy()
    else:
        raise

