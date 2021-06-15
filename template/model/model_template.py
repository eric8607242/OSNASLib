import torch.nn as nn

from ..base_model import BaseModel
from ..block_builder import get_block


class {{customize_class}}Model(BaseModel):
    def _construct_stage_layers(self, architecture, bn_momentum, bn_track_running_stats, *args, **kwargs):
        """ Construct searched layers in entire search stage.

        Return:
            stages (nn.Sequential)
        """
        return stages


