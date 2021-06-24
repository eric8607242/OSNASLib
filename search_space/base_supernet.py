import math

import numpy as np

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_builder import get_block

class BaseSuperlayer(nn.Module):
    """ The abstract class of the layer of the supernet """
    def __init__(self, micro_cfg, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats, *args, **kwargs):
        super(BaseSuperlayer, self).__init__()
        self.micro_cfg = micro_cfg

        self._construct_supernet_layer(in_channels, out_channels, stride, bn_momentum, bn_track_running_stats)

    @abstractmethod
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats, *args, **kwargs):
        """ Construct the supernet layer module.
        """
        raise NotImplemented
    
    def forward(self, x):
        state_forward = getattr(self, f"_{self.forward_state}_forward")
        y = state_forward(x)

        return y

    # Single-path NAS
    def set_activate_architecture(self, architecture):
        """ Activate the path based on the architecture. Utilizing in single-path NAS.

        Args:
            architecture (torch.tensor): The block index for each layer.
        """
        self.architecture = architecture

    # Differentaible NAS
    def set_arch_param(self, arch_param):
        """ Set architecture parameter directly

        Args:
            arch_param (torch.tensor)
        """
        self.arch_param = arch_param
        

    def initialize_arch_param(self, device):
        """ Initialize architecture parameters for differentiable based search strategy.
        
        Args:
            device (torch.device)
        """
        micro_len = len(self.micro_cfg)

        self.arch_param = nn.Parameter(
            1e-3 * torch.randn((1, micro_len), requires_grad=False, device=device))


    def get_arch_param(self):
        """ Return architecture parameters.

        Return:
            (list)
        """
        return [self.arch_param]


    def get_best_arch_param(self):
        """ Get the best neural architecture from architecture parameters (argmax).

        Return:
            best_architecture (np.ndarray)
        """
        best_architecture = self.arch_param.data.argmax(dim=1)
        best_architecture = best_architecture.cpu().numpy().tolist()

        return best_architecture

    def set_forward_state(self, state, **kwargs):
        """ Set supernet forward state. ["gumbel_softmax", "softmax", "single", "sum"]

        Args:
            state (str): The state in model forward.
        """
        self.forward_state = state
        if "tau" in kwargs:
            self.tau = kwargs["tau"]

    def _gumbel_softmax_forward(self, x):
        weight = F.gumbel_softmax(self.arch_param[0], dim=0, tau=self.tau) 
        y = sum(p * b(x) for p, b in zip(weight, self.supernet_layer))
        return y

    def _softmax_forward(self, x):
        weight = F.softmax(self.arch_param[0], dim=0)
        y = sum(p * b(x) for p, b in zip(weight, self.supernet_layer))
        return y

    def _single_forward(self, x):
        y = self.supernet_layer[self.architecture](x)
        return y

    def _sum_forward(self, x):
        weight = self.arch_param[0]
        y = sum(p * b(x) for p, b in zip(weight, self.supernet_layer))
        return y


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
        y = x
        for i, l in enumerate(self.first_stages):
            y = l(y)

        for i, l in enumerate(self.search_stages):
            y = l(y)

        for i, l in enumerate(self.last_stages):
            y = l(y)

        return y

    @abstractmethod
    def get_model_cfg(self, classes):
        raise NotImplemented

    @abstractmethod
    def get_model_cfg_shape(self):
        raise NotImplemented
    

    def set_arch_param(self, arch_param):
        """ Set architecture parameter directly

        Args:
            arch_param (torch.tensor)
        """
        arch_param = arch_param.reshape(len(self.search_stages), 1, -1)
        for ap, l in zip(arch_param, self.search_stages):
            l.set_arch_param(ap)
        

    def set_activate_architecture(self, architecture):
        """ Activate the path based on the architecture. Utilizing in single-path NAS.

        Args:
            architecture (torch.tensor): The block index for each layer.
        """
        architecture = architecture.reshape(len(self.search_stages), -1)
        for a, l in zip(architecture, self.search_stages):
            l.set_activate_architecture(a)


    def initialize_arch_param(self, device):
        """ Initialize architecture parameters for differentiable for each layer.
        
        Args:
            device (torch.device)
        """
        for l in self.search_stages:
            l.initialize_arch_param(device)

    def get_arch_param(self):
        """ Return architecture parameters.

        Return:
            arch_param (torch.tensor)
            arch_param_list (list)
        """
        arch_param_list = []
        for l in self.search_stages:
            arch_param_list.extend(l.get_arch_param())

        return torch.cat(arch_param_list), arch_param_list

    def get_best_arch_param(self):
        """ Get the best neural architecture from architecture parameters (argmax).

        Return:
            best_architecture (np.ndarray)
        """
        best_architecture_list = []
        for l in self.search_stages:
            best_architecture_list.extend(l.get_best_arch_param())
        
        #best_architecture = np.concatenate(best_architecture_list, axis=-1)
        best_architecture = np.array(best_architecture_list)
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
