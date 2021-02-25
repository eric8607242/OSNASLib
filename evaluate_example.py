from config_file.arg_config import *
from lib.container import Evaluator


if __name__ == "__main__":
    args = get_init_config("evaluate")

    evaluator = Evaluator(args)
    evaluator.evaluate()
    evaluator.inference()
