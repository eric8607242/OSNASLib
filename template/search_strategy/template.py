from .base import BaseSearcher

class {{customize_class}}Searcher(BaseSearcher):
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger):
        super({{customize_class}}, self).__init__(config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger)

    def step(self):
        """ The searcher step before each iteration. 
        """
        pass

    def search(self):
        """ Searching the best architecture based on the hardware constraints and the supernet.

        Return:
            best_architecture (np.ndarray)
            best_architecture_hc (float)
            best_architecture_top1 (float)
        """
        return best_architecture, best_architecture_hc, best_architecture_top1

