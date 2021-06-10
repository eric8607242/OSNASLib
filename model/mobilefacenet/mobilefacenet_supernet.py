import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSupernet, BaseSuperlayer
from ..block_builder import get_block

class MobileFaceNetSuperlayer(BaseSuperlayer):
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats):
        supernet_layer = nn.ModuleList()
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
            supernet_layer.append(block)
        return supernet_layer

    def forward(self, x):
        if self.forward_state == "gumbel_sum":
            weight = F.gumbel_softmax(self.arch_param[i], dim=0) 
            x = sum(p * b(x) for p, b in zip(weight, self.supernet_layer))

        elif self.forward_state == "sum":
            weight = F.softmax(self.arch_param[i], dim=0)
            x = sum(p * b(x) for p, b in zip(weight, self.supernet_layer))

        elif self.forward_state == "single":
            x = self.supernet_layer[self.architecture](x)
        return x

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
        

    def initialize_arch_param(self):
        micro_len = len(self.micro_cfg)

        self.arch_param = nn.Parameter(
            1e-3 * torch.randn((1, micro_len), requires_grad=False, device=next(self.parameters()).device))


    def get_arch_param(self):
        """ Return architecture parameters.

        Return:
            self.arch_param (nn.Parameter)
        """
        return self.arch_param


    def get_best_arch_param(self):
        """ Get the best neural architecture from architecture parameters (argmax).

        Return:
            best_architecture (np.ndarray)
        """
        best_architecture = self.arch_param.data.argmax(dim=1)
        best_architecture = best_architecture.cpu().numpy()

        return best_architecture


    def set_forward_state(self, state):
        """ Set supernet forward state. ["single", "sum"]

        Args:
            state (str): The state in model forward.
        """
        self.forward_state = state

class MobileFaceNetSupernet(BaseSupernet):
    superlayer_builder = MobileFaceNetSuperlayer

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
        return macro_cfg, micro_cfg

    def get_model_cfg_shape(self):
        """ Return the shape of model config for the architecture generator.

        Return 
            model_cfg_shape (Tuple)
        """
        return (len(macro_cfg["search"]), len(micro_cfg))
