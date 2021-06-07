from .base import BaseSampler

class {{customize_class}}Sampler(BaseSampler):
    """ Sampler is the strategy to decide how to train each candidate block of the supernet.
    """
    def __init__(self, micro_len, macro_len, model):
        super({{customize_class}}, self).__init__(micro_len, macro_len, model)

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
        pass
