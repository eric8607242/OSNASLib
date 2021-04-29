import time

import torch
import torch.nn as nn

from .base_agent import MetaAgent

from utils import get_optimizer, get_lr_scheduler, resume_checkpoint
from model import Model, load_architecture

class EvaluateAgent(MetaAgent):
    def __init__(self, config):
        super(EvaluateAgent, self).__init__(config, "evaluate")

        # Construct model and correspond optimizer ======================================
        architecture = load_architecture(self.config.searched_model_path)

        self.model = Model(
            self.macro_cfg,
            self.micro_cfg,
            architecture,
            self.config.classes,
            self.config.dataset)

        self.optimizer = get_optimizer(
            self.model.parameters(),
            self.config.optimizer,
            learning_rate=self.config.lr,
            weight_decay=self.config.weight_decay,
            logger=self.logger,
            momentum=self.config.momentum,
            alpha=self.config.alpha,
            beta=self.config.beta)

        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            self.config.lr_scheduler,
            self.logger,
            step_per_epoch=len(
                self.train_loader),
            step_size=self.config.decay_step,
            decay_ratio=self.config.decay_ratio,
            total_epochs=self.config.epochs)
        # =================================================================================

        # Resume checkpoint ===============================================================
        self.start_epoch = 0
        if self.config.resume:
            self.start_epoch = resume_checkpoint(
                self.model,
                self.config.resume,
                self.optimizer,
                self.lr_scheduler)
            logger.info(
                "Resume training from {} at epoch {}".format(
                    self.config.resume, start_epoch))
        # =================================================================================

        if device.type == "cuda" and self.config.ngpu >= 1:
            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(
                self.model, list(range(self.config.ngpu)))

    def fit(self):
        self.evaluate()
        self.inference()

    def evaluate(self):
        start_time = time.time()
        self.train_loop(
            self.model,
            self.train_loader,
            self.test_loader,
            self.optimizer,
            self.lr_scheduler,
            self.start_epoch)
        logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))

    def inference(self):
        start_time = time.time()
        top1_acc = self.validate(self.model, self.test_loader, 0)
        logger.info("Final Top1 accuracy : {}".format(top1_acc))
        logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))
