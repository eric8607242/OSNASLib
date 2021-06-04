import time
import numpy as np

import torch
import torch.nn as nn

from .base import BaseSearcher

class {{customize_class}}Searcher(BaseSearcher):
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger):
        super({{customize_class}}, self).__init__(config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger)



    def search(self):
        return best_architecture, best_architecture_hc, best_architecture_top1

