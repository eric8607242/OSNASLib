from .training_agent import CFTrainingAgent
from ..base_agent import MetaSearchAgent, MetaEvaluateAgent

class CFSearchAgent(MetaSearchAgent):
    agent_state = "search"
    training_agent = CFTrainingAgent()

class CFEvaluateAgent(MetaEvaluateAgent):
    agent_state = "evaluate"
    training_agent = CFTrainingAgent()
