import torch
import torch.nn as nn

from .base import BaseSampler

class FairnessSampler(BaseSampler):
    def __init__(self, model):
        super(FairnessSampler, self).__init__(model)
        
        # The order for training candidate blocks in each layer.
        self.architecture_order = []
        for macro_len, micro_len in self.search_space_cfg_shape:
            self.architecture_order.extend([[i for i in range(micro_len)] * macro_len])

        for sub_architecture_order in self.architecture_order:
            random.shuffle(sub_architecture_order)

        # The index of candidate blocks in each layer for training.
        self.architecture_micro_index = []
        for macro_len, micro_len in self.search_space_cfg_shape:
            self.architecture_micro_index.extend([0 for i in range(macro_len)])

        # The length of candidate blocks in each layer.
        self.architecture_micro_len = []
        for macro_len, micro_len in self.search_space_cfg_shape:
            self.architecture_micro_len.extend([micro_len for i in range(macro_len)])


    def step(self):
        """ The sampler step before each iteration

        In each step, the sampler should decide the strategy to update the supernet.
        We provide four protocal in the supernet:
            gumbel_softmax: In each layer, the architecture parameter will be passed into the Gumbel Softmax layer
                            to transform into the weight summation as 1. After weights transformation, the supernet
                            will weighted sum the output of each canididate block. Therefore, user should pass the 
                            hyperparameter `tau` as the argument during `set_forward_state`

            softmax: In each layer, the architecture parameter will be passed into the Softmax layer
                     to transform into the weight summation as 1. After weights transformation, the supernet
                     will weighted sum the output of each canididate block.

            sum:     In each layer, the supernet will weighted sum the output of each candidate block.
                     Therefore, user shoud utilize architecture parameters or set the architecture parameters
                     with supernet.set_arch_param(arch_param)

            single:  In each layer, the supernet will only forward one of all candidate blocks.
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
        architecture = [0 for i in range(self.total_macro_cfg_len))
        for i, a in enumerate(architecture):
            block_index = self.architecture_order[i][self.architecture_micro_index[i]]
            architecture[i] = block_index

            self.architecture_micro_index[i] += 1
            if self.architecture_micro_index[i] == self.architecture_micro_len[i]:
                self.architecture_micro_index[i] = 0
                random.shuffle(self.architecture_order[i])
        return architecture
