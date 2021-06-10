import math

import numpy as np

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_builder import get_block

class BaseSuperlayer(nn.Module):
    """ The abstract class of the layer of the supernet """
    def __init__(self, micro_cfg, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats):
        super(BaseSuperlayer, self).__init__()
        self.micro_cfg = micro_cfg
        self.forward_state = None

        self.supernet_layer = self._construct_supernet_layer(in_channels, out_channels, stride, bn_momentum, bn_track_running_stats)

    @abstractmethod
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats):
        return supernet_layer
    
    @abstractmethod
    def forward(self, x):
        return x

    # Single-path NAS
    @abstractmethod
    def set_activate_architecture(self, architecture):
        """ Activate the path based on the architecture. Utilizing in single-path NAS.

        Args:
            architecture (torch.tensor): The block index for each layer.
        """
        raise NotImplemented

    # Differentaible NAS
    @abstractmethod
    def set_arch_param(self, arch_param):
        """ Set architecture parameter directly

        Args:
            arch_param (torch.tensor)
        """
        raise NotImplemented
        
    @abstractmethod
    def initialize_arch_param(self):
        raise NotImplemented

    @abstractmethod
    def get_arch_param(self):
        """ Return architecture parameters.

        Return:
            self.arch_param (nn.Parameter)
        """
        raise NotImplemented

    @abstractmethod
    def get_best_arch_param(self):
        """ Get the best neural architecture from architecture parameters (argmax).

        Return:
            best_architecture (np.ndarray)
        """
        raise NotImplemented

    @abstractmethod
    def set_forward_state(self, state):
        """ Set supernet forward state. ["single", "sum"]

        Args:
            state (str): The state in model forward.
        """
        raise NotImplemented
        self.forward_state = state



class BaseSupernet(nn.Module):
    """ The abstract class of supernet.
    """
    def __init__(
            self,
            classes,
            dataset,
            bn_momentum=0.1,
            bn_track_running_stats=True):
        super(BaseSupernet, self).__init__()
        self.macro_cfg, self.micro_cfg = self.get_model_cfg(classes)

        self.classes = classes
        self.dataset = dataset

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
        self.search_stages = nn.ModuleList()
        for l_cfg in self.macro_cfg["search"]:
            in_channels, out_channels, stride = l_cfg

            layer = self.superlayer_builder(
                    micro_cfg=self.micro_cfg,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bn_momentum=bn_momentum,
                    bn_track_running_stats=bn_track_running_stats)
            self.search_stages.append(layer)

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

    def forward(self, x):
        for i, l in enumerate(self.first_stages):
            x = l(x)

        for i, l in enumerate(self.search_stages):
            x = l(x)

        for i, l in enumerate(self.last_stages):
            x = l(x)

        return x
    

    def set_arch_param(self, arch_param):
        """ Set architecture parameter directly

        Args:
            arch_param (torch.tensor)
        """
        arch_param = arch_param.reshape(len(self.search_stages), -1)
        for ap, l in zip(arch_param, self.search_stages):
            l.set_arch_param(ap)
        

    def set_activate_architecture(self, architecture):
        """ Activate the path based on the architecture. Utilizing in single-path NAS.

        Args:
            architecture (torch.tensor): The block index for each layer.
        """
        architecture = architecture.reshape(len(self.search_stages))
        for a, l in zip(architecture, self.search_stages):
            l.set_activate_architecture(a)


    def initialize_arch_param(self):
        for l in self.search_stages:
            l.initialize_arch_param()

    def get_arch_param(self):
        """ Return architecture parameters.

        Return:
            arch_param (torch.tensor)
            arch_param_list (list)
        """
        arch_param_list = []
        for l in self.search_stages:
            arch_param_list.append(l.get_arch_param())

        return torch.cat(arch_param_list), arch_param_list

    def get_best_arch_param(self):
        """ Get the best neural architecture from architecture parameters (argmax).

        Return:
            best_architecture (np.ndarray)
        """
        best_architecture_list = []
        for l in self.search_stages:
            best_architecture_list.append(l.get_best_arch_param())
        
        best_architecture = np.concatenate(best_architecture_list, axis=-1)
        return best_architecture

    def set_forward_state(self, state):
        """ Set supernet forward state. ["single", "sum"]

        Args:
            state (str): The state in model forward.
        """
        for l in self.search_stages:
            l.set_forward_state(state)


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

    @abstractmethod
    def get_model_cfg(self, classes):
        raise NotImplemented

    @abstractmethod
    def get_model_cfg_shape(self):
        raise NotImplemented
