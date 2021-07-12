import os
import time

import torch

from utils import AverageMeter, save
from ..base_training_agent import MetaTrainingAgent

class {{customize_class}}TrainingAgent(MetaTrainingAgent):
    """The training agent to train the supernet and the searched architecture.

    By implementing TrainingAgent class, users can adapt the searching and evaluating agent into
    various tasks easily.
    """
    @staticmethod
    def searching_evaluate(model, val_loader, device, criterion):
        """ Evaluating the performance of the supernet. The search strategy will evaluate
        the architectures by this static method to search.

        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            device (torch.device)
            criterion (nn.Module)

        Return:
            evaluate_metric (float): The performance of the supernet.
        """
        model.eval()
        with torch.no_grad():
            for step, (X, y) in enumerate(val_loader):
                raise NotImplemented
        return evaluate_metric


    def _training_step(self, model, train_loader, agent, epoch, print_freq=100):
        """
        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)
        """
        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(train_loader):
            if agent.agent_state == "search":
                agent._iteration_preprocess()

        raise NotImplemented

    def _validate_step(self, model, val_loader, agent, epoch):
        """
        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)

        Return:
            evaluate_metric (float): The performance of the searched model.
        """
        model.eval()
        start_time = time.time()
        if agent.agent_state == "search":
            agent._iteration_preprocess()
        raise NotImplemented
