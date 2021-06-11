import torch
import torch.nn as nn

from ..base import BaseSupernet, BaseSuperlayer
from ..block_builder import get_block
from .sgnas_utils import UnifiedSubBlock

class SGNASSuperlayer(BaseSuperlayer):
    """ The unified block in SGNAS
    """
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats):
        """ Construct the supernet layer module.
        """
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.kernel_size_list = []
        for b_cfg in self.micro_cfg:
            self.kernel_size_list.append(b_cfg[1])

        self.max_expansion_rate = self.micro_cfg[0][-1]["max_expansion_rate"]
        self.min_expansion_rate = self.micro_cfg[0][-1]["min_expansion_rate"]

        hidden_channel = int(in_channels * self.max_expansion_rate)

        self.point_wise = get_block(block_type="conv",
                              in_channels=in_channels,
                              out_channels=hidden_channel,
                              kernel_size=1,
                              stride=1,
                              activation=activation,
                              se=se,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **kwargs
                              )


        # Unified Block ===================================
        self.block_in_channels = hidden_channel // self.max_expansion_rate
        self.block_out_channels = hidden_channel // self.max_expansion_rate

        self.unified_block = nn.ModuleList()
        for b in range(self.max_expansion_rate):
            block = UnifiedSubBlock(micro_cfg=self.micro_cfg,
                                    in_channels=self.block_in_channels,
                                    out_channels=self.block_out_channels,
                                    stride=stride,
                                    bn_momentum=bn_momentum,
                                    bn_track_running_stats=bn_track_running_stats)

            self.unified_block.append(block)
        # ====================================================

        self.point_wise_1 = get_block(block_type="conv",
                              in_channels=hidden_channel,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              activation=None,
                              se=se,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **{"bn": False}
                              )

        self.sbn = nn.ModuleList()
        for i in range(self.max_expansion_rate-self.min_expansion_rate+1):
            self.sbn.append(nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = self.point_wise(x)

        # Unified block ====================
        split_y = torch.split(y, self.block_in_channels, dim=1)
        y = torch.cat([block_i(y_i) for y_i, block_i in zip(split_y, self.unified_block)], dim=1)
        # ==================================

        y = self.point_wise_1(y)
        y = self.sbn[self.sbn_index](y)

        y = y + x if self.use_res_connect else y

        return y


    # Single-path NAS
    def set_activate_architecture(self, architecture):
        """ Activate the path based on the architecture. Utilizing in single-path NAS.

        Args:
            architecture (torch.tensor): The block index for each layer.
        """
        self.sbn_index = 0
        split_architecture = torch.split(architecture, len(self.kernel_size_list)) 

        for block, a in zip(self.unified_block, split_architecture):
            # Force sampling
            a = a.sort()[0]
            if "skip" in self.kernel_size_list:
                self.sbn_index += a[self.kernel_size_list.index("skip")]
            block.set_activate_architecture(a)

    # Differentaible NAS
    def set_arch_param(self, arch_param):
        """ Set architecture parameter directly

        Args:
            arch_param (torch.tensor)
        """
        self.sbn_index = 0
        split_arch_param = torch.split(arch_param, len(self.kernel_size_list))

        for block, ap in zip(self.unified_block, split_arch_param):
            if 0 in self.kernel_size_list:
                self.sbn_index += ap[self.kernel_size_list.index(0)]

            block.set_arch_param(ap)
        self.sbn_index = round(self.sbn_index)
        
    def initialize_arch_param(self, device):
        """ Initialize architecture parameters for differentiable based search strategy.
        
        Args:
            device (torch.device)
        """
        for block in self.unified_block:
            block.initialize_arch_param()

    def get_arch_param(self):
        """ Return architecture parameters.

        Return:
            arch_param_list (list)
        """
        self.sbn_index = 0

        arch_param_list = []
        for block in self.unified_block:
            arch_param = block.get_arch_param()
            if 0 in self.kernel_size_list:
                self.sbn_index += arch_param[self.kernel_size_list.index(0]
                
            arch_param_list.append(arch_param)
        self.sbn_index = round(self.sbn_index)

        return arch_param_list

    def get_best_arch_param(self):
        """ Get the best neural architecture from architecture parameters (argmax).

        Return:
            best_architecture (np.ndarray)
        """
        best_architecture_list = []
        for block in self.unified_block:
            best_architecture_list.append(block.get_best_arch_param())

        return best_architecture_list



class SGNASSupernet(BaseSupernet):
    superlayer_builder = SGNASSuperlayer

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
        micro_cfg = [["conv", 3, False, "relu", {"max_expansion_rate": 6, "min_expansion_rate": 2}],
                    ["conv", 5, False, "relu", {"max_expansion_rate": 6, "min_expansion_rate": 2}],
                    ["conv", 7, False, "relu", {"max_expansion_rate": 6, "min_expansion_rate": 2}],
                    ["skip", 0, False, None, {"max_expansion_rate": 6, "min_expansion_rate": 2}]]

        macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["conv", 3, 32, 2, 3, "relu", False, {}],  # stride 1 for CIFAR
                      ["mobile", 32, 16, 1, 3, "relu", False, {"expansion_rate": 1}]],
            # in_channels, out_channels, stride
            "search": [[16, 24, 2],  # stride 1 for CIFAR
                       [24, 24, 1],
                       [24, 24, 1],
                       [24, 24, 1],
                       [24, 40, 2],
                       [40, 40, 1],
                       [40, 40, 1],
                       [40, 40, 1],
                       [40, 80, 2],
                       [80, 80, 1],
                       [80, 80, 1],
                       [80, 80, 1],
                       [80, 96, 1],
                       [96, 96, 1],
                       [96, 96, 1],
                       [96, 96, 1],
                       [96, 192, 2],
                       [192, 192, 1],
                       [192, 192, 1],
                       [192, 192, 1],
                       [192, 320, 1]],
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "last": [["conv", 320, 1280, 1, 1, "relu", False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}],
                     ["classifier", 1280, classes, 0, 0, 0, 0, {}]]
        }
        return macro_cfg, micro_cfg

    def get_model_cfg_shape(self):
        """ Return the shape of model config for the architecture generator.

        Return 
            model_cfg_shape (Tuple)
        """
        return (len(macro_cfg["search"])*len(self.micro_cfg), len(self.micro_cfg))
