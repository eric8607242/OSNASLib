import abc

class BaseSampler:
    def __init__(self, micro_len, macro_len, model):
        self.micro_len = micro_len
        self.macro_len = macro_len
        
        self.model = model

    @abc.abstractmethod
    def step(self):
        return NotImplemented

    def get_block_len(self):
        """ Get the number of the candidate blocks.
        Return:
            self.micro_len (int)
        """
        return self.micro_len

