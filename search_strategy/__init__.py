import sys

from .random_search import RandomSearcher
from .evolution_search import EvolutionSearcher
from .differentiable import DifferentiableSearcher

def get_search_strategy(name):
    return getattr(sys.modules[__name__], name)
