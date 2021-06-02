import abc 

import numpy as np

import torch
import torch.nn as nn

from utils import AverageMeter, accuracy
from agent import get_agent_cls

class BaseSearcher:
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, logger):
        self.config = config
        self.logger = logger

        self.supernet = supernet
        self.val_loader = val_loader

        self.lookup_table = lookup_table
        self.training_strategy = training_strategy

        self.device = device
        self.logger = logger

        self.target_hc = self.config["target_hc"]
        self.top1 = AverageMeter()

    def step(self):
        pass

    @abc.abstractmethod
    def search(self):
        return NotImplemented

    def _evaluate(self):
        self.supernet.eval()

        agent_cls = get_agent_cls(self.config["agent"]["main_agent"])
        acc_info = agent_cls.validate_step(self.supernet, self.val_loader, self.device, self.criterion)
        return acc_info[0]

    def evaluate_architectures(self, architectures_list):
        architectures_top1_acc = []
        for i, architecture in enumerate(architectures_list):
            self.supernet.module.set_activate_architecture(architecture) if isinstance(
                self.supernet, nn.DataParallel) else self.supernet.set_activate_architecture(architecture)

            top1_avg = self._evaluate()
            architectures_top1_acc.append(top1_avg)
            self.logger.info(f"Evaluate {i} architecture top1-avg : {100*top1_avg}%")

        architectures_top1_acc = np.array(architectures_top1_acc)
        return architectures_top1_acc

