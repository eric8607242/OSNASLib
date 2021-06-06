from .training_agent import {{customize_class}}TrainingAgent
from ..base_agent import MetaSearchAgent, MetaEvaluateAgent

class {{customize_class}}SearchAgent(MetaSearchAgent):
    agent_state = "search"
    training_agent = {{customize_class}}TrainingAgent()

class {{customize_class}}EvaluateAgent(MetaEvaluateAgent):
    agent_state = "evaluate"
    training_agent = {{customize_class}}TrainingAgent()
