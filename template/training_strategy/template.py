import torch
import torch.nn as nn

from .base import BaseSampler

class {{customize_name}}(BaseSampler):
    def __init__(self, micro_len, macro_len, model):
        super({{customize_name}}, self).__init__(micro_len, macro_len, model)

    def step(self):
        pass
