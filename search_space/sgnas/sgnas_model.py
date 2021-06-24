import numpy as np

import torch.nn as nn

from ..base_model import BaseModel
from ..block_builder import get_block

class SGNASModel(BaseModel):
    def _construct_stage_layers(self, architecture, bn_momentum, bn_track_running_stats, *args, **kwargs):
        """ Construct searched layers in entire search stage.

        Return:
            stages (nn.Sequential)
        """
        split_architecture = np.split(architecture, len(self.macro_cfg["search"]))

        stages = []
        for l_cfg, block_idxs in zip(self.macro_cfg["search"], split_architecture):
            in_channels, out_channels, stride = l_cfg

            kernel_size_list = []
            for block_idx in block_idxs:
                block_type, kernel_size, se, activation, kwargs = self.micro_cfg[block_idx]
                if kernel_size == 0:
                    continue
                kernel_size_list.append(kernel_size)

            layer = get_block(block_type="mixconv",
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation="relu",
                              se=False,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **{"kernel_size_list": kernel_size_list,
                                 "expansion_rate": len(kernel_size_list)})
            stages.append(layer)

        stages = nn.Sequential(*stages)
        return stages


