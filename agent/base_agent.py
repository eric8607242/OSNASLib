import os
import time

import torch

from utils import get_logger, get_writer, set_random_seed, AverageMeter, accuracy, save
from criterion import get_criterion
from dataflow import get_dataloader

class MetaAgent:
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

        # Init agent
        getattr(self, self.agent_state)()


    def search_agent(self):
        # Construct model and correspond optimizer ======================================
        supernet = self._construct_supernet()
        self.supernet = supernet.to(self.device)

        self.macro_cfg, self.micro_cfg = self.supernet.get_model_cfg()
        self._optimizer_init(self.supernet)

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
        self.search_strategy = search_strategy_class(
                self.config, 
                self.supernet, 
                self.val_loader, 
                self.lookup_table, 
                self.training_strategy, 
                self.device, self.logger)

        # Resume checkpoint ===============================================================
        self._resume(self.supernet)
        self.supernet = self._parallel_process(self.supernet)


    def evaluate_agent(self):
        # Construct model and correspond optimizer ======================================
        architecture = load_architecture(self.config["experiment_path"]["searched_model_path"])

        supernet = self._construct_supernet()
        self.macro_cfg, self.micro_cfg = supernet.get_model_cfg()

        model = Model(
            self.macro_cfg,
            self.micro_cfg,
            architecture,
            self.config["dataset"]["classes"],
            self.config["dataset"]["dataset"])
        self.model = model.to(self.device)

        self._optimizer_init(self.model)
        # =================================================================================

        # Resume checkpoint ===============================================================
        self._resume(self.model)
        self.model = self._parallel_process(self.model)

    @abstractmethod
    def fit(self):
        raise NotImplementedError


    @abstractmethod
    def train_loop(self, model,
                         train_loader,
                         val_loader,
                         optimizer,
                         lr_scheduler):
        raise NotImplementedError

    @abstractmethod
    def _training_step(
            self,
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch):
        raise NotImplementedError


    @abstractmethod
    def validate(self, model, val_loader, epoch, inference=True):
        raise NotImplementedError


    @abstractmethod
    def _epoch_stats_logging(
            self,
            start_time,
            epoch,
            val_or_train,
            info_for_writer=""):
        raise NotImplementedError


    @abstractmethod
    def _intermediate_stats_logging(
            self,
            outs,
            y,
            loss,
            step,
            epoch,
            N,
            len_loader,
            val_or_train,
            print_freq=100):
        raise NotImplementedError


    @abstractmethod
    def _reset_average_tracker(self):
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
        if self.config["train"]["resume"]:
            self.start_epochs = resume_checkpoint(
                    model,
                    self.config["experiment_path"]["resume_path"],
                    self.optimizer,
                    self.lr_scheduler)
            self.logger.info(
                "Resume training from {} at epoch {}".format(
                    self.config["experiment_path"]["resume_path"], self.start_epochs))

    def _parallel_process(self, model):
        if self.device.type == "cuda" and self.config["train"]["ngpu"] >= 1:
            return nn.DataParallel(
                model, list(range(self.config["train"]["ngpu"])))

    def _construct_supernet(self):
        supernet_class = get_supernet(self.config["agent"]["supernet_agent"])
        supernet = supernet_class(
            self.config["dataset"]["classes"],
            self.config["dataset"]["dataset"],
            self.config["search_utility"]["search_strategy"],
            bn_momentum=self.config["train"]["bn_momentum"],
            bn_track_running_stats=self.config["train"]["bn_track_running_stats"])
        return supernet
