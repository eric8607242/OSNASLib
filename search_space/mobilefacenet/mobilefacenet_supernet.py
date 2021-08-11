import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_supernet import BaseSupernet, BaseSuperlayer
from ..block_builder import get_block

class MobileFaceNetSuperlayer(BaseSuperlayer):
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats, *args, **kwargs):
        """ Construct the supernet layer module.
        """
        self.supernet_layer = nn.ModuleList()
        for b_cfg in self.micro_cfg:
            block_type, kernel_size, se, activation, cfg_kwargs = b_cfg
            block = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **cfg_kwargs
                              )
            self.supernet_layer.append(block)


class MobileFaceNetSupernet(BaseSupernet):
    superlayer_builder = MobileFaceNetSuperlayer

    @staticmethod
    def get_seach_space_cfg(classes):
        """ Return the macro and micro configurations of the search space.

        Args:
            classes (int): The number of output class (dimension).
        
        Return:
            search_space_cfg (list): list of search space tuple, there are macro_cfg and micro_cfg in each tuple.
                macro_cfg (dict): The structure of the entire supernet. The structure is split into three parts, "first", "search", "last"
                micro_cfg (list): The all configurations in each layer of supernet.
        """
        # block_type, kernel_size, se, activation, kwargs
        micro_cfg = [["mobile", 3, False, "prelu", {"expansion_rate": 1}],
                           ["mobile", 3, False, "prelu", {"expansion_rate": 2}],
                           ["mobile", 3, False, "prelu", {"expansion_rate": 4}],
                           ["mobile", 5, False, "prelu", {"expansion_rate": 1}],
                           ["mobile", 5, False, "prelu", {"expansion_rate": 2}],
                           ["mobile", 5, False, "prelu", {"expansion_rate": 4}],
                           ["skip", 0, False, "prelu", {}]]
        macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["conv", 3, 64, 2, 3, "prelu", False, {}],
                ["conv", 64, 64, 1, 3, "prelu", False, {"group":64}]],  # stride 1 for CIFAR
            # in_channels, out_channels, stride
            "search": [[64, 64, 2],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 128, 2],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 2],
                       [128, 128, 1],
                       [128, 128, 1]],
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "last": [
                     ["conv", 128, 512, 1, 1, "prelu", False, {}],
                     ["conv", 512, 512, 1, 7, None, False, {"group":512}],
                     ["conv", 512, classes, 1, 1, None, False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}]]
        }
        search_space_cfg = [(macro_cfg, micro_cfg)]
        return search_space_cfg

    def get_seach_space_cfg_shape(self):
        """ Return the shape of search space config for the architecture generator.

        Return 
            search_space_cfg_shape (list)
        """
        search_space_cfg_shape = [(len(macro_cfg), len(micro_cfg)) for macro_cfg, micro_cfg in self.search_space_cfg]
        return search_space_cfg_shape
