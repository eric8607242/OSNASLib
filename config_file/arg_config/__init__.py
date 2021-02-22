import os 

from .search_config import get_search_config
from .evaluate_config import get_evaluate_config

def get_init_config(config_state):
    if config_state == "search":
        args = get_search_config()
    elif config_state == "evaluate":
        args = get_evaluate_config()

    args = setting_path_config(args)

    return args


def setting_path_config(args):
    """
    Concatenate root path to each path argument
    """
    if not os.path.exists(os.path.join(args.root_path, args.title+"_{}".format(args.random_seed))):
        args.root_path = os.path.join(args.root_path, args.title+"_{}".format(args.random_seed))
        os.makedirs(args.root_path)

    args.lookup_table_path = os.path.join(args.root_path, args.lookup_table_path)
    args.supernet_model_path = os.path.join(args.root_path, args.supernet_model_path)
    args.searched_model_path = os.path.join(args.root_path, args.searched_model_path)

    return args

