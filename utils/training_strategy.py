import random

import torch

class TrainingStrategy:
    """
    Support multiple sampling strategy ["uniform", "fair"]
    """
    def __init__(self, sample_strategy, micro_len, macro_len):
        self.micro_len = micro_len
        self.macro_len = macro_len
        self.sample_strategy = sample_strategy

        if self.sample_strategy == "uniform":
            pass
        elif self.sample_strategy == "fair":
            self.architecture_order = [[i for i in range(self.micro_len)] for j in range(self.macro_len)]
            for sub_architecture_order in self.architecture_order:
                random.shuffle(sub_architecture_order)

            self.architecture_index = [0 for i in range(self.macro_len)]
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


    def step(self):
        if self.sample_strategy == "uniform":
            pass
        elif self.sample_strategy == "fair":
            pass
        elif self.sample_strategy == "differentiable":
            pass
        else:
            raise 

    def get_uniform_architectures(self):
        architecture = torch.randint(low=0, high=self.micro_len, size=(self.macro_len,))
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


