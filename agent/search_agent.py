import time

import torch
import torch.nn as nn

from .base_agent import MetaAgent

from utils import get_optimizer, get_lr_scheduler, resume_checkpoint
from model import save_architecture, LookUpTable, get_supernet

from search_strategy import get_search_strategy
from training_strategy import get_training_strategy

class SearchAgent(MetaAgent):
    def __init__(self, config, title):
        super(SearchAgent, self).__init__(config, title, "search")

        # Construct model and correspond optimizer ======================================
        supernet_class = get_supernet(self.config["agent"]["supernet_agent"])
        supernet = supernet_class(
            self.config["dataset"]["classes"],
            self.config["dataset"]["dataset"],
            self.config["search_utility"]["search_strategy"],
            bn_momentum=self.config["train"]["bn_momentum"],
            bn_track_running_stats=self.config["train"]["bn_track_running_stats"])
        self.supernet = supernet.to(self.device)

        self.macro_cfg, self.micro_cfg = self.supernet.get_model_cfg()

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


        # Construct search utility ========================================================
        training_strategy_class = get_training_strategy(self.config["agent"]["training_strategy_agent"])
        self.training_strategy = training_strategy_class(len(self.micro_cfg), len(self.macro_cfg["search"]), self.supernet)

        self.lookup_table = LookUpTable(
            self.macro_cfg,
            self.micro_cfg,
            self.config["experiment_path"]["lookup_table_path"],
            self.config["dataset"]["input_size"],
            info_metric=self.config["search_utility"]["info_metric"])

        search_strategy_class = get_search_strategy(self.config["agent"]["search_strategy_agent"])
        self.search_strategy = search_strategy_class(self.config, self.supernet, self.val_loader, self.lookup_table, self.training_strategy, self.device, self.logger)
        # =================================================================================


        # Resume checkpoint ===============================================================
        if self.config["train"]["resume"]:
            self.start_epochs = resume_checkpoint(
                    self.supernet,
                    self.config["experiment_path"]["resume_path"],
                    self.optimizer,
                    self.lr_scheduler)
            self.logger.info(
                "Resume training from {} at epoch {}".format(
                    self.config["experiment_path"]["resume_path"], self.start_epochs))
        # =================================================================================

        if self.device.type == "cuda" and self.config["train"]["ngpu"] >= 1:
            self.supernet = nn.DataParallel(
                self.supernet, list(range(self.config["train"]["ngpu"])))

    def fit(self):
        self.search()

    def search(self):
        start_time = time.time()
        self.logger.info("Searching process start!")

        if not self.config["search_utility"]["directly_search"]:
            self.train_loop(
                self.supernet,
                self.train_loader,
                self.val_loader,
                self.optimizer,
                self.lr_scheduler)

        best_architecture, best_architecture_hc, best_architecture_top1 = self.search_strategy.search()

        self.logger.info("Best architectrue : {}".format(best_architecture))
        self.logger.info(
            "Best architectrue top1 : {:.3f}".format(
                best_architecture_top1 * 100))
        self.logger.info("Best architectrue hc : {}".format(best_architecture_hc))

        save_architecture(self.config["experiment_path"]["searched_model_path"], best_architecture)
        self.logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))
