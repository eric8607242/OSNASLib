import sys

from .evaluate_agent import EvaluateAgent
from .search_agent import SearchAgent

def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
