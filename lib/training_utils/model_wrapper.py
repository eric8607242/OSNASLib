import torch.nn as nn

class ModelWrapper:
    """
    Training supernet with different protocal. (e.g., Uniform, Fairstrict, and differentaible)
    """
    def __init__(self, model, training_strategy, sample_strategy, g_optimizer=None):
        self.sample_strategy = sample_strategy
        self.training_strategy = training_strategy

        self.g_optimizer = g_optimizer

        self.model = model

    def step(self):
        if self.sample_strategy == "uniform":
            architecture = self.generate_training_architecture()
            self.model.module.set_activate_architecture(architecture) if isinstance(self.model, nn.DataParallel) else self.model.set_activate_architecture(architecture)

        elif self.sample_strategy == "fair":
            architecture = self.training_strategy.get_fair_architectures()
            self.model.module.set_activate_architecture(architecture) if isinstance(self.model, nn.DataParallel) else self.model.set_activate_architecture(architecture)

        elif self.sample_strategy == "differentiable":
            self.g_optimizer.step()
            
        else:
            raise

