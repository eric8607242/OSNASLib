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
        """ The sampler step before each iteration

        In each step, the sampler should decide the strategy to update the supernet.
        We provide two protocal in the supernet:
            sum:    In each layer, the supernet will weighted sum the output of each candidate block.
                    Therefore, user shoud utilize architecture parameters or set the architecture parameters
                    with supernet.set_arch_param(arch_param)
            single: In each layer, the supernet will only forward one of all candidate blocks.
                    Therefore, user should set the activate block in each layer 
                    by supernet.set_activate_architecture(architecture)

        User should set the protocal to sum or single by supernet.set_forward_state(state).
        """
        self.model.module.set_forward_state("single") if isinstance(
            self.model, nn.DataParallel) else self.model.set_forward_state("single")

        architecture = self.generate_training_architecture()
        self.model.module.set_activate_architecture(architecture) if isinstance(
            self.model, nn.DataParallel) else self.model.set_activate_architecture(architecture)


    def generate_training_architecture(self):
        """ Generate the architecture to activate into the supernet.

        Return:
            architecture (torch.Tensor)
        """
        architecture = torch.zeros(self.macro_len, dtype=torch.int)
        for i, a in enumerate(architecture):
            block_index = self.architecture_order[i][self.architecture_index[i]]
            architecture[i] = block_index

            self.architecture_index[i] += 1
            if self.architecture_index[i] == self.micro_len:
                self.architecture_index[i] = 0
                random.shuffle(self.architecture_order[i])
        return architecture
