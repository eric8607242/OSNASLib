from .base import BaseSampler

class {{customize_class}}Sampler(BaseSampler):
    def __init__(self, micro_len, macro_len, model):
        super({{customize_class}}, self).__init__(micro_len, macro_len, model)

    def step(self):
        pass
