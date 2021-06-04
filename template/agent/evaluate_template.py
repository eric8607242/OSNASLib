import time

from .base_agent import {{customize_class}}MetaAgent

class {{customize_class}}EvaluateAgent({{customize_class}}MetaAgent):
    """{{customize_class}} evaluate agent
    """
    agent_state = "evaluate_agent"
    def fit(self):
        self._evaluate()
        self._inference()

    def _iteration_preprocess(self):
        pass

    def _evaluate(self):
        pass

    def _inference(self):
        pass
