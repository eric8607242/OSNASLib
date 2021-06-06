import abc 

import numpy as np

import torch
import torch.nn as nn

from utils import AverageMeter, accuracy

class BaseSearcher:
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

    def _evaluate(self):
        from agent import get_agent_cls
        self.supernet.eval()

        agent_cls = get_agent_cls(self.config["agent"]["main_agent"])
        acc_info = agent_cls.training_agent.searching_evaluate(self.supernet, self.val_loader, self.device, self.criterion)
        return acc_info[0]

    def evaluate_architectures(self, architectures_list):
        architectures_acc = []
        for i, architecture in enumerate(architectures_list):
            self.supernet.module.set_activate_architecture(architecture) if isinstance(
                self.supernet, nn.DataParallel) else self.supernet.set_activate_architecture(architecture)

            acc = self._evaluate()
            architectures_acc.append(acc)
            self.logger.info(f"Evaluate {i} architecture acc-avg : {acc}")

        architectures_acc = np.array(architectures_acc)
        return architectures_acc

