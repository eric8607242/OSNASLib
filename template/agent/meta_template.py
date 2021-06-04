import os
import time

import torch

from utils import AverageMeter, accuracy, save

from ..base_agent import MetaAgent

class {{customize_name}}MetaAgent(MetaAgent):
    """{{customize_name}} meta agent
    """
    evaluate_metric = ""
    def _training_step(
            self,
            model,
            train_loader,
            epoch,
            print_freq=100):
        pass


    def _validate(self, model, val_loader, epoch):
        pass

    @staticmethod
    def searching_evaluate(model, val_loader, device, criterion):
        pass

