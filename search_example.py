from config import get_init_config
from container import SearchAgent


if __name__ == "__main__":
    args = get_init_config("search")

    searcher = SearchAgent(args)
    searcher.search()
