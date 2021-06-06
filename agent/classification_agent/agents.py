from .training_agent import CFTrainingAgent
from ..base_search_agent import MetaSearchAgent
from ..base_evaluate_agent import MetaEvaluateAgent

class CFSearchAgent(MetaSearchAgent):
    agent_state = "search"
    training_agent = CFTrainingAgent()

class CFEvaluateAgent(MetaEvaluateAgent):
    agent_state = "evaluate"
    training_agent = CFTrainingAgent()
