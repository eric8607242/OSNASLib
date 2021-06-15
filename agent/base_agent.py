import os
import time

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from utils import get_logger, get_writer, set_random_seed, get_optimizer, get_lr_scheduler, resume_checkpoint
from model import calculate_model_efficient
from criterion import get_criterion
from dataflow import get_dataloader

class MetaAgent:
    """ The abstract class for the agent of each class. 
    Initialization for searching or evaluating agent.
    """
    def __init__(self, config, title):
        self.config = config

        self.logger = get_logger(config["logs_path"]["logger_path"])
        self.writer = get_writer(
            title,
            config["train"]["random_seed"],
            config["logs_path"]["writer_path"])

        if self.config["train"]["random_seed"] is not None:
            set_random_seed(config["train"]["random_seed"])

        self.device = torch.device(config["train"]["device"])

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.config["dataset"]["dataset"], 
                                                                self.config["dataset"]["dataset_path"], 
                                                                self.config["dataset"]["input_size"],
                                                                self.config["dataset"]["batch_size"], 
                                                                self.config["dataset"]["num_workers"], 
                                                                train_portion=self.config["dataset"]["train_portion"])

        self.criterion = get_criterion(config["agent"]["criterion_agent"], config["criterion"])

        self.epochs = config["train"]["epochs"]
        self.start_epochs = 0

        self.config = config
        
        self._init_agent_state()

    @abstractmethod
    def fit(self):
        """ Fit agent for searching or evaluating.
        """
        raise NotImplementedError

    def _optimizer_init(self, model):
        self.optimizer = get_optimizer(
            model.parameters(),
            self.config["optim"]["optimizer"],
            learning_rate=self.config["optim"]["lr"],
            weight_decay=self.config["optim"]["weight_decay"],
            logger=self.logger,
            momentum=self.config["optim"]["momentum"],
            alpha=self.config["optim"]["alpha"],
            beta=self.config["optim"]["beta"])

        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            self.config["optim"]["scheduler"],
            self.logger,
            step_per_epoch=len(
                self.train_loader),
            step_size=self.config["optim"]["decay_step"],
            decay_ratio=self.config["optim"]["decay_ratio"],
            total_epochs=self.config["train"]["epochs"])

    def _resume(self, model):
        """ Load the checkpoint of model, optimizer, and lr scheduler.
        """
        if self.config["train"]["resume"]:
            self.start_epochs = resume_checkpoint(
                    model,
                    self.config["experiment_path"]["resume_path"],
                    None,
                    None)
            self.logger.info(
                "Resume training from {} at epoch {}".format(
                    self.config["experiment_path"]["resume_path"], self.start_epochs))

    def _parallel_process(self, model):
        if self.device.type == "cuda" and self.config["train"]["ngpu"] >= 1:
            return nn.DataParallel(
                model, list(range(self.config["train"]["ngpu"])))
        else:
            return model
