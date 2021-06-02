import sys

from .classification_agent import CFSearchAgent, CFEvaluateAgent


def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
