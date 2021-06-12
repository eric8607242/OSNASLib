import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSuperlayer
from ..block_builder import get_block


class ZeroConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ZeroConv, self).__init__()
        self.zero_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
    
    def forward(self, x):
        return self.zero_conv(x) * 0


class UnifiedSubBlock(BaseSuperlayer):
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats):
        self.supernet_layer = nn.ModuleList()
        # Micro confg is  kernel size list
        for i, b_cfg in enumerate(self.micro_cfg):
            block_type, kernel_size, se, activation, kwargs = b_cfg
            if kernel_size == 0:
                operation = ZeroConv(in_channels=in_channels,
                                     out_channels=out_channels,
                                     stride=stride)

            else:
                operation = get_block(block_type=block_type,
                                  in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  activation=activation,
                                  se=se,
                                  bn_momentum=bn_momentum,
                                  bn_track_running_stats=bn_track_running_stats,
                                  **kwargs)
            self.supernet_layer.append(operation)
