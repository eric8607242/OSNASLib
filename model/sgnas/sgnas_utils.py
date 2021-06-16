import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_supernet import BaseSuperlayer
from ..block_builder import get_block


class UnifiedSubBlock(BaseSuperlayer):
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats, *args, **kwargs):
        self.supernet_layer = nn.ModuleList()
        # Micro confg is  kernel size list
        for i, b_cfg in enumerate(self.micro_cfg):
            block_type, kernel_size, se, activation, cfg_kwargs = b_cfg
            operation = get_block(block_type=block_type,
                                  in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  activation=activation,
                                  se=se,
                                  bn_momentum=bn_momentum,
                                  bn_track_running_stats=bn_track_running_stats,
                                  **cfg_kwargs)
            self.supernet_layer.append(operation)
