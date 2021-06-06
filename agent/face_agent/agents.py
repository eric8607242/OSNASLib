from .training_agent import FRTrainingAgent
from ..base_agent import MetaSearchAgent, MetaEvaluateAgent

class FRSearchAgent(MetaSearchAgent):
    agent_state = "search"
    training_agent = FRTrainingAgent()

class FREvaluateAgent(MetaEvaluateAgent):
    agent_state = "evaluate"
    training_agent = FRTrainingAgent()
