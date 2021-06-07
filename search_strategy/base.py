import abc 

import numpy as np

import torch
import torch.nn as nn

from utils import AverageMeter, accuracy

class BaseSearcher:
    """The abstract class for each search strategy.
    """
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger):
        self.config = config
        self.logger = logger

        self.supernet = supernet
        self.val_loader = val_loader

        self.lookup_table = lookup_table
        self.training_strategy = training_strategy

        self.criterion = criterion
        
        self.device = device
        self.logger = logger

        self.target_hc = self.config["target_hc"]

    @abc.abstractmethod
    def step(self):
        return NotImplemented

    @abc.abstractmethod
    def search(self):
        return NotImplemented

    def evaluate_architectures(self, architectures_list):
        """Evaluate architectures list to get the performance of each architecture.

        Args:
            architectures_list (list): List of architectures

        Return:
            architectures_acc (list): List of the performance of each architecture in architectures_list
        """
        from agent import get_agent_cls

        architectures_acc = []
        for i, architecture in enumerate(architectures_list):
            self.supernet.module.set_activate_architecture(architecture) if isinstance(
                self.supernet, nn.DataParallel) else self.supernet.set_activate_architecture(architecture)

            agent_cls = get_agent_cls(self.config["agent"]["main_agent"])
            acc_info = agent_cls.training_agent.searching_evaluate(self.supernet, self.val_loader, self.device, self.criterion)

            architectures_acc.append(acc_info[0])
            self.logger.info(f"Evaluate {i} architecture acc-avg : {acc}")

        architectures_acc = np.array(architectures_acc)
        return architectures_acc

