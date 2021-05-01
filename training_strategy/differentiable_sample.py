import torch
import torch.nn as nn

from .base import BaseSampler

class DifferentiableSampler(BaseSampler):
    def step(self):
        self.model.module.set_forward_state("sum") if isinstance(
            self.model, nn.DataParallel) else self.model.set_forward_state("sum")


