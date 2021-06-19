import math

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_builder import get_block


class BaseModel(nn.Module):
    def __init__(
            self,
            macro_cfg,
            micro_cfg,
            architecture,
            bn_momentum=0.1,
            bn_track_running_stats=True):
        super(BaseModel, self).__init__()
        self.micro_cfg = micro_cfg
        self.macro_cfg = macro_cfg

        # First Stage
        self.first_stages = nn.ModuleList()
        for l_cfg in self.macro_cfg["first"]:
            block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs = l_cfg

            layer = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **kwargs)
            self.first_stages.append(layer)

        # Search Stage
        self.stages = self._construct_stage_layers(architecture, bn_momentum, bn_track_running_stats)

        # Last Stage
        self.last_stages = nn.ModuleList()
        for l_cfg in self.macro_cfg["last"]:
            block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs = l_cfg

            layer = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **kwargs)
            self.last_stages.append(layer)

        self._initialize_weights()

    @abstractmethod
    def _construct_stage_layers(self, architecture, bn_momentum, bn_track_running_statsm *args, **kwargs):
        """ Construct searched layers in entire search stage.

        Return:
            stages (nn.Sequential)
        """
        return stages


    def forward(self, x):
        y = x
        for i, l in enumerate(self.first_stages):
            y = l(y)

        y = self.stages(y)

        for i, l in enumerate(self.last_stages):
            y = l(y)

        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
