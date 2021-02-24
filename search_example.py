from lib.config_file.arg_config import *
from lib.container import Searcher


if __name__ == "__main__":
    args = get_init_config("search")

    searcher = Searcher(args)
    searcher.search()
