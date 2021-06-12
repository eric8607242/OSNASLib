import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSupernet, BaseSuperlayer
from ..block_builder import get_block

class SPOSSuperlayer(BaseSuperlayer):
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats):
        """ Construct the supernet layer module.
        """
        self.supernet_layer = nn.ModuleList()
        for b_cfg in self.micro_cfg:
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
            self.supernet_layer.append(block)


class SPOSSupernet(BaseSupernet):
    superlayer_builder = SPOSSuperlayer

    @staticmethod
    def get_model_cfg(classes):
        """ Return the macro and micro configurations of the search space.

        Args:
            classes (int): The number of output class (dimension).
        
        Return:
            macro_cfg (dict): The structure of the entire supernet. The structure is split into three parts, "first", "search", "last"
            micro_cfg (list): The all configurations in each layer of supernet.
        """
        # block_type, kernel_size, se, activation, kwargs
        micro_cfg = [["shuffle", 3, False, "relu", {}],
                          ["shuffle", 5, False, "relu", {}],
                          ["shuffle", 7, False, "relu", {}],
                          ["shuffleX", 0, False, "relu", {}]]
        macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["conv", 3, 16, 2, 3, "relu", False, {}]],  # stride 1 for CIFAR
            # in_channels, out_channels, stride
            "search": [[16, 64, 2],  # stride 1 for CIFAR
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 160, 2],
                       [160, 160, 1],
                       [160, 160, 1],
                       [160, 160, 1],
                       [160, 320, 2],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 640, 2],
                       [640, 640, 1],
                       [640, 640, 1],
                       [640, 640, 1]],
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "last": [["conv", 640, 1024, 1, 1, "relu", False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}],
                     ["classifier", 1024, classes, 0, 0, 0, 0, {}]]
        }
        return macro_cfg, micro_cfg

    def get_model_cfg_shape(self):
        """ Return the shape of model config for the architecture generator.

        Return 
            model_cfg_shape (Tuple)
        """
        return (len(self.macro_cfg["search"]), len(self.micro_cfg))
