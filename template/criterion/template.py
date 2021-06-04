import torch
import torch.nn as nn
import torch.nn.functional as F

class {{customize_name}}(nn.Module):
    def __init__(self, criterion_config):
        super({{customize_name}}, self).__init__()
        self.criterion_config = criterion_config

    def forward(self, x, target):
        # Write your code here
        return loss
