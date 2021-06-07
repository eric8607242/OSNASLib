from .training_agent import FRTrainingAgent
from ..base_search_agent import MetaSearchAgent
from ..base_evaluate_agent import MetaEvaluateAgent

class FRSearchAgent(MetaSearchAgent):
    agent_state = "search"
    training_agent = FRTrainingAgent()

class FREvaluateAgent(MetaEvaluateAgent):
    agent_state = "evaluate"
    training_agent = FRTrainingAgent()
