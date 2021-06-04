import time

from .base_agent import {{customize_name}}MetaAgent

class {{customize_name}}EvaluateAgent({{customize_name}}):
    """{{customize_name}} evaluate agent
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
