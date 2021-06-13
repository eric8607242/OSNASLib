import abc

import torch.nn as nn

class BaseSampler:
    def __init__(self, model):
        self.macro_len, self.micro_len = model.module.get_model_cfg_shape() if isinstance(model, nn.DataParallel) \
                    else model.get_model_cfg_shape()
        
        self.model = model

    @abc.abstractmethod
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
        raise NotImplemented

    def get_block_len(self):
        """ Get the number of the candidate blocks.
        Return:
            self.micro_len (int)
        """
        return self.micro_len

