import torch
import torch.nn as nn

from .base import BaseSampler

class FairnessSampler(BaseSampler):
    def __init__(self, micro_len, macro_len, model):
        super(FairnessSampler, self).__init__(micro_len, macro_len, model)
        
        self.architecture_order = [
            [i for i in range(self.micro_len)] for j in range(self.macro_len)]
        for sub_architecture_order in self.architecture_order:
            random.shuffle(sub_architecture_order)

        self.architecture_index = [0 for i in range(self.macro_len)]


    def step(self):
        self.model.module.set_forward_state("single") if isinstance(
            self.model, nn.DataParallel) else self.model.set_forward_state("single")

        architecture = self.generate_training_architecture()
        self.model.module.set_activate_architecture(architecture) if isinstance(
            self.model, nn.DataParallel) else self.model.set_activate_architecture(architecture)


    def generate_training_architecture(self):
        architecture = torch.zeros(self.macro_len, dtype=torch.int)
        for i, a in enumerate(architecture):
            block_index = self.architecture_order[i][self.architecture_index[i]]
            architecture[i] = block_index

            self.architecture_index[i] += 1
            if self.architecture_index[i] == self.micro_len:
                self.architecture_index[i] = 0
                random.shuffle(self.architecture_order[i])
        return architecture
