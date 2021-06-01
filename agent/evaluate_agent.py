import time

import torch
import torch.nn as nn

from .base_agent import MetaAgent

from utils import get_optimizer, get_lr_scheduler, resume_checkpoint
from model import Model, load_architecture, get_supernet

class EvaluateAgent(MetaAgent):
    def __init__(self, config):
        super(EvaluateAgent, self).__init__(config, "evaluate")

        # Construct model and correspond optimizer ======================================
        architecture = load_architecture(self.config.searched_model_path)

        supernet_class = get_supernet(self.config["agent"]["supernet_agent"])
        supernet = supernet_class(
            self.config["dataset"]["classes"],
            self.config["dataset"]["dataset"],
            self.config["search_utility"]["search_strategy"],
            bn_momentum=self.config["train"]["bn_momentum"],
            bn_track_running_stats=self.config["train"]["bn_track_running_stats"])
        self.macro_cfg, self.micro_cfg = self.supernet.get_model_cfg()

        model = Model(
            self.macro_cfg,
            self.micro_cfg,
            architecture,
            self.config["dataset"]["classes"]
            self.config["dataset"]["dataset"])
        self.model = model.to(self.device)

        self.optimizer = get_optimizer(
            self.supernet.parameters(),
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
        # =================================================================================

        # Resume checkpoint ===============================================================
        if self.config["train"]["resume"]:
            self.start_epochs = resume_checkpoint(
                    self.model,
                    self.config["experiment_path"]["resume_path"],
                    self.optimizer,
                    self.lr_scheduler)
            self.logger.info(
                "Resume training from {} at epoch {}".format(
                    self.config["experiment_path"]["resume_path"], self.start_epochs))
        # =================================================================================

        if self.device.type == "cuda" and self.config["train"]["ngpu"] >= 1:
            self.model = nn.DataParallel(
                self.model, list(range(self.config["train"]["ngpu"])))

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
