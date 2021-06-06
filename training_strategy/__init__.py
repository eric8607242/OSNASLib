import sys

def get_training_strategy(name):
    return getattr(sys.modules[__name__], name)

from .differentiable_sample import DifferentiableSampler
from .uniform_sample import UniformSampler
from .fair_sample import FairnessSampler
