import math

import torch
import torch.nn as nn

from utils.network_utils import get_block

def construct_supernet_layer(micro_cfg, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats):
    supernet_layer = nn.ModuleList()
    for b_cfg in micro_cfg:
        block_type, kernel_size, se, activation, kwargs = b_cfg
        block = get_block(block_type=block_type,
                          in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          activation=activation,
                          se=se,
                          bn_momentum=bn_momentum,
                          bn_track_running_stats=bn_track_running_stats,
                          **kwargs
                        )
        supernet_layer.append(block)
    return supernet_layer

class Supernet(nn.Module):
    def __init__(self, supernet_cfg, micro_cfg, classes, dataset, bn_momentum=0.1, bn_track_running_stats=True):
        super(Supernet, self).__init__()
        self.classes = classes
        self.dataset = dataset

        bn_momentum = bn_momentum
        bn_track_running_stats = bn_track_running_stats

        # First Stage
        self.first_stages = nn.ModuleList()
        for l_cfg in supernet_cfg["first"]:
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
        self.search_stages= nn.ModuleList()
        for l_cfg in supernet_cfg["search"]:
            in_channels, out_channels, stride = l_cfg

            layer = construct_supernet_layer(micro_cfg=micro_cfg,
                                             in_channels=in_channels,
                                             out_channels=out_channels,
                                             stride=stride,
                                             bn_momentum=bn_momentum,
                                             bn_track_running_stats=bn_track_running_stats)
            self.search_stages.append(layer)

        # Last Stage
        self.last_stages = nn.ModuleList()
        for l_cfg in supernet_cfg["last"]:
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

    def forward(self, x):
        for i, l in enumerate(self.first_stages):
            x = l(x) 

        for i, l in enumerate(self.search_stages):
            x = l(x)

        for i, l in enumerate(self.last_stages):
            x = l(x)

        return x

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

