import time

from model import save_architecture

from .base_agent import {{customize_name}}MetaAgent

class {{customize_name}}SearchAgent({{customize_name}}):
    """ {{customize_name}} search agent
    """
    agent_state = "search_agent"
    def fit(self):
        self._search()

    def _iteration_preprocess(self):
        pass

    def _search(self):
        pass


