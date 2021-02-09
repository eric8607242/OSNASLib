from .random_search import random_search
from .evolution_search import evoluation_algorithm


class SearchStrategy:
    def __init__(self, search_strategy):
        self.search_strategy = search_strategy

    def search(self, trainer, training_strategy, supernet, val_loader, lookup_table, args, logger):
        if self.search_strategy == "evolution":
            best_architecture = evoluation_algorithm(trainer, training_strategy, supernet, val_loader, lookup_table, args.target_hc, logger, generation_num=args.generation_num, population=args.population, parent_num=args.parent_num, info_metric=args.info_metric)
        elif self.search_strategy == "random_search":
            best_architecture = random_search(trainer, training_strategy, supernet, val_loader, lookup_table, args.target_hc, logger, random_iteration=args.random_iteration, info_metric=args.info_metric)
        elif self.search_strategy == "differentaible":
            best_architecture = supernet.get_best_architecture()

        else:
            raise 

        return best_architecture
