from config import get_init_config
from container import EvaluateAgent


if __name__ == "__main__":
    args = get_init_config("evaluate")

    evaluator = EvaluateAgent(args)
    evaluator.evaluate()
    evaluator.inference()
