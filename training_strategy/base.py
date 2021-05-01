
class BaseSampler:
    def __init__(self, micro_len, macro_len, model):
        self.micro_len = micro_len
        self.macro_len = macro_len
        
        self.model = model

    def step(self):
        pass

    def get_block_len(self):
        return self.micro_len

    def generate_training_architecture(self):
        pass
