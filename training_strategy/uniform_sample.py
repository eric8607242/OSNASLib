import torch
import torch.nn as nn

from .base import BaseSampler


class UniformSampler(BaseSampler):
    def step(self):
        self.model.module.set_forward_state("single") if isinstance(
            self.model, nn.DataParallel) else self.model.set_forward_state("single")

        architecture = self.generate_training_architecture()
        self.model.module.set_activate_architecture(architecture) if isinstance(
            self.model, nn.DataParallel) else self.model.set_activate_architecture(architecture)

    def generate_training_architecture(self):
        architecture = torch.randint(
            low=0, high=self.micro_len, size=(self.macro_len,))
        return architecture


