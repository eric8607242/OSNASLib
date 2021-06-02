import time

import torch
import torch.nn as nn

from .base_agent import CFMetaAgent

from utils import get_optimizer, get_lr_scheduler, resume_checkpoint
from model import Model, load_architecture, get_supernet

class CFEvaluateAgent(CFMetaAgent):
    """Classification evaluate agent
    """
    agent_state = "evaluate_agent"
    def fit(self):
        self.evaluate()
        self.inference()

    def evaluate(self):
        start_time = time.time()
        self.logger.info("Evaluating process start!")

        self.train_loop(
            self.model,
            self.train_loader,
            self.test_loader,
            self.optimizer,
            self.lr_scheduler)
        self.logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))

    def inference(self):
        start_time = time.time()
        top1_acc = self.validate(self.model, self.test_loader, 0)
        self.logger.info("Final Top1 accuracy : {}".format(top1_acc))
        self.logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))
