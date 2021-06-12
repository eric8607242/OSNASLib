import abc

import torch.nn as nn

class BaseSampler:
    def __init__(self, model):
        self.macro_len, self.micro_len = model.module.get_model_cfg_shape() if isinstance(model, nn.DataParallel) \
                    else model.get_model_cfg_shape()
        
        self.model = model

    @abc.abstractmethod
    def step(self):
        return NotImplemented

    def get_block_len(self):
        """ Get the number of the candidate blocks.
        Return:
            self.micro_len (int)
        """
        return self.micro_len

