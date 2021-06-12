import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSupernet, BaseSuperlayer
from ..block_builder import get_block

class FBNetSuperlayer(BaseSuperlayer):
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


class FBNetSSupernet(BaseSupernet):
    superlayer_builder = FBNetSuperlayer

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
        micro_cfg = [["mobile", 3, False, "relu", {"expansion_rate": 1, "point_group": 1}],
                           ["mobile", 3, False, "relu", {
                               "expansion_rate": 1, "point_group": 2}],
                           ["mobile", 3, False, "relu", {
                               "expansion_rate": 3, "point_group": 1}],
                           ["mobile", 3, False, "relu", {
                               "expansion_rate": 6, "point_group": 1}],
                           ["mobile", 5, False, "relu", {
                               "expansion_rate": 1, "point_group": 1}],
                           ["mobile", 5, False, "relu", {
                               "expansion_rate": 1, "point_group": 2}],
                           ["mobile", 5, False, "relu", {
                               "expansion_rate": 3, "point_group": 1}],
                           ["mobile", 5, False, "relu", {
                               "expansion_rate": 6, "point_group": 1}],
                           ["skip", 0, False, "relu", {}]]
        macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["conv", 3, 16, 2, 3, "relu", False, {}]],  # stride 1 for CIFAR
            # in_channels, out_channels, stride
            "search": [[16, 16, 1],
                       [16, 24, 2],  # stride 1 for CIFAR
                       [24, 24, 1],
                       [24, 24, 1],
                       [24, 24, 1],
                       [24, 32, 2],
                       [32, 32, 1],
                       [32, 32, 1],
                       [32, 32, 1],
                       [32, 64, 2],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 112, 1],
                       [112, 112, 1],
                       [112, 112, 1],
                       [112, 112, 1],
                       [112, 184, 2],
                       [184, 184, 1],
                       [184, 184, 1],
                       [184, 184, 1],
                       [184, 352, 1]],
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "last": [["conv", 352, 1504, 1, 1, "relu", False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}],
                     ["classifier", 1504, classes, 0, 0, 0, 0, {}]]
        }
        return macro_cfg, micro_cfg

    def get_model_cfg_shape(self):
        """ Return the shape of model config for the architecture generator.

        Return 
            model_cfg_shape (Tuple)
        """
        return (len(self.macro_cfg["search"]), len(self.micro_cfg))


class FBNetLSupernet(BaseSupernet):
    superlayer_builder = FBNetSuperlayer

    @staticmethod
    def get_model_cfg(classes):
        micro_cfg = [["mobile", 3, False, "relu", {"expansion_rate": 1, "point_group": 1}],
                           ["mobile", 3, False, "relu", {
                               "expansion_rate": 1, "point_group": 2}],
                           ["mobile", 3, False, "relu", {
                               "expansion_rate": 3, "point_group": 1}],
                           ["mobile", 3, False, "relu", {
                               "expansion_rate": 6, "point_group": 1}],
                           ["mobile", 5, False, "relu", {
                               "expansion_rate": 1, "point_group": 1}],
                           ["mobile", 5, False, "relu", {
                               "expansion_rate": 1, "point_group": 2}],
                           ["mobile", 5, False, "relu", {
                               "expansion_rate": 3, "point_group": 1}],
                           ["mobile", 5, False, "relu", {
                               "expansion_rate": 6, "point_group": 1}],
                           ["skip", 0, False, "relu", {}]]
        macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["conv", 3, 16, 2, 3, "relu", False, {}]],  # stride 1 for CIFAR
            # in_channels, out_channels, stride
            "search": [[16, 16, 1],
                       [16, 24, 2],  # stride 1 for CIFAR
                       [24, 24, 1],
                       [24, 24, 1],
                       [24, 24, 1],
                       [24, 32, 2],
                       [32, 32, 1],
                       [32, 32, 1],
                       [32, 32, 1],
                       [32, 64, 2],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 112, 1],
                       [112, 112, 1],
                       [112, 112, 1],
                       [112, 112, 1],
                       [112, 184, 2],
                       [184, 184, 1],
                       [184, 184, 1],
                       [184, 184, 1],
                       [184, 352, 1]],
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "last": [["conv", 352, 1984, 1, 1, "relu", False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}],
                     ["classifier", 1984, classes, 0, 0, 0, 0, {}]]
        }
        return macro_cfg, micro_cfg

    def get_model_cfg_shape(self):
        """ Return the shape of model config for the architecture generator.

        Return 
            model_cfg_shape (Tuple)
        """
        return (len(self.macro_cfg["search"]), len(self.micro_cfg))
