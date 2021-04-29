import random

import torch
import torch.nn as nn


class TrainingStrategy:
    """
    Support multiple sampling strategy ["uniform", "fair", "differentiable"]
    """

    def __init__(self, sample_strategy, micro_len, macro_len, model):
        self.micro_len = micro_len
        self.macro_len = macro_len
        self.sample_strategy = sample_strategy

        self.model = model

        if self.sample_strategy == "uniform":
            pass
        elif self.sample_strategy == "fair":
            self.architecture_order = [
                [i for i in range(self.micro_len)] for j in range(self.macro_len)]
            for sub_architecture_order in self.architecture_order:
                random.shuffle(sub_architecture_order)

            self.architecture_index = [0 for i in range(self.macro_len)]
        elif self.sample_strategy == "differentiable":
            pass
        else:
            raise

    def step(self):
        if self.sample_strategy == "uniform":
            self.model.module.set_forward_state("single") if isinstance(
                self.model, nn.DataParallel) else self.model.set_forward_state("single")

            architecture = self.generate_training_architecture()
            self.model.module.set_activate_architecture(architecture) if isinstance(
                self.model, nn.DataParallel) else self.model.set_activate_architecture(architecture)

        elif self.sample_strategy == "fair":
            self.model.module.set_forward_state("single") if isinstance(
                self.model, nn.DataParallel) else self.model.set_forward_state("single")

            architecture = self.get_fair_architectures()
            self.model.module.set_activate_architecture(architecture) if isinstance(
                self.model, nn.DataParallel) else self.model.set_activate_architecture(architecture)

        elif self.sample_strategy == "differentiable":
            self.model.module.set_forward_state("sum") if isinstance(
                self.model, nn.DataParallel) else self.model.set_forward_state("sum")

        else:
            raise

    def generate_training_architecture(self):
        if self.sample_strategy == "uniform":
            architecture = self.get_uniform_architectures()
        elif self.sample_strategy == "fair":
            architecture = self.get_fair_architectures()
        elif self.sample_strategy == "differentiable":
            pass
        else:
            raise

        return architecture

    def get_uniform_architectures(self):
        architecture = torch.randint(
            low=0, high=self.micro_len, size=(
                self.macro_len,))
        return architecture

    def get_fair_architectures(self):
        architecture = torch.zeros(self.macro_len, dtype=torch.int)
        for i, a in enumerate(architecture):
            block_index = self.architecture_order[i][self.architecture_index[i]]
            architecture[i] = block_index

            self.architecture_index[i] += 1
            if self.architecture_index[i] == self.micro_len:
                self.architecture_index[i] = 0
                random.shuffle(self.architecture_order[i])
        return architecture

    def get_block_len(self):
        return self.micro_len
