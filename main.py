import os
import argparse
import yaml
from pprint import pprint
from agent import get_agent_cls

from config import process_config


def main(config_path, title):
    with open(config_path) as f:
        config = yaml.full_load(f)
        config = process_config(config, title)
        pprint(config)

    agent_cls = get_agent_cls(config["agent"]["main_agent"])
    agent = agent_cls(config, title)
    agent.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    args = vars(parser.parse_args())
    main(args["config"], args["title"])


