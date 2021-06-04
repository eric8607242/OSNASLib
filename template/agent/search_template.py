import time

from model import save_architecture

from .base_agent import {{customize_class}}MetaAgent

class {{customize_class}}SearchAgent({{customize_class}}MetaAgent):
    """ {{customize_class}} search agent
    """
    agent_state = "search_agent"
    def fit(self):
        self._search()

    def _iteration_preprocess(self):
        pass

    def _search(self):
        pass


