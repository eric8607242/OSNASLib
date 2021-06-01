import os

def process_config(config, title):
    root_path = os.path.join(config["logs_path"]["logger_path"], title)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    for k, v in config["experiment_path"].items():
        config["experiment_path"][k] = os.path.join(root_path, v)

    if not os.path.exists(config["experiment_path"]["checkpoint_root_path"]):
        os.makedirs(config["experiment_path"]["checkpoint_root_path"])

    return config
