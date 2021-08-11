import numpy as np

import torch.nn as nn

from .base import BaseSampler


class UniformSampler(BaseSampler):
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
            architecture (list)
        """

        architecture = []
        for macro_len, micro_len in self.search_space_cfg_shape:
            architecture.append(np.random.randint(low=0, high=micro_len, size=(macro_len,))

        return architecture


