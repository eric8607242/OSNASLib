import os
import time

import torch

from utils import get_logger, get_writer, set_random_seed, AverageMeter, accuracy, save
from criterion import get_criterion
from dataflow import get_dataloader

class MetaAgent:
    def __init__(self, config, title, agent_state="search"):
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
                                                                self.config["dataset"]["batch_size"], 
                                                                self.config["dataset"]["num_workers"], 
                                                                train_portion=self.config["dataset"]["train_portion"])

        self.criterion = get_criterion(config["agent"]["criterion_agent"], config["criterion"])

        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()

        self.agent_state = agent_state

        self.epochs = config["train"]["epochs"]
        self.start_epochs = 0

        self.config = config


    def train_loop(self, model,
                         train_loader,
                         val_loader,
                         optimizer,
                         lr_scheduler):
        """
        Support mode:
        1) Two stages training:
            - Uniform sampling
            - Strict sampling
        2) Jointly training:
            - Training architecture parameter and supernet
        """
        best_top1_acc = 0.0

        for epoch in range(self.start_epochs, self.epochs):
            self.logger.info("Start to train for epoch {}".format(epoch))
            self.logger.info(
                "Learning Rate : {:.8f}".format(
                    optimizer.param_groups[0]["lr"]))

            self._training_step(
                model,
                train_loader,
                optimizer,
                lr_scheduler,
                epoch)
            val_top1 = self.validate(
                model,
                val_loader,
                epoch,
                inference=False if self.agent_state == "search" else True)

            if val_top1 > best_top1_acc:
                self.logger.info(
                    "Best validation top1-acc : {}. Save model!".format(val_top1 * 100))
                best_top1_acc = val_top1
                save(
                    model,
                    self.config["experiment_path"]["best_checkpoint_path"],
                    optimizer,
                    lr_scheduler,
                    epoch + 1)

            save(
                model,
                os.path.join(
                    self.config["experiment_path"]["checkpoint_root_path"],
                    "{}_{}.pth".format(
                        self.agent_state,
                        epoch)),
                optimizer,
                lr_scheduler,
                epoch + 1)

    def _training_step(
            self,
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch):
        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(train_loader):

            if self.agent_state == "search":
                # If search strategy is "differentiable". Update architecture
                # parameter.
                self.search_strategy.step()

            X, y = X.to(
                self.device, non_blocking=True), y.to(
                self.device, non_blocking=True)
            N = X.shape[0]

            optimizer.zero_grad()

            if self.agent_state == "search":
                self.training_strategy.step()

            outs = model(X)

            loss = self.criterion(outs, y)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            self._intermediate_stats_logging(
                outs,
                y,
                loss,
                step,
                epoch,
                N,
                len_loader=len(train_loader),
                val_or_train="Train")

        self._epoch_stats_logging(start_time, epoch, val_or_train="Train")
        self._reset_average_tracker()

    def validate(self, model, val_loader, epoch, inference=True):
        """
            inference(bool) : True, evaluate specific architecture.
                              False, evaluate supernet.
        """
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(val_loader):
                X, y = X.to(
                    self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True)
                N = X.shape[0]

                if not inference:
                    self.training_strategy.step()

                outs = model(X)

                loss = self.criterion(outs, y)
                self._intermediate_stats_logging(
                    outs,
                    y,
                    loss,
                    step,
                    epoch,
                    N,
                    len_loader=len(val_loader),
                    val_or_train="Valid")

        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time, epoch, val_or_train="Valid")
        self._reset_average_tracker()

        return top1_avg

    def _epoch_stats_logging(
            self,
            start_time,
            epoch,
            val_or_train,
            info_for_writer=""):
        """
        Logging training information and record information in Tensorboard for each epoch
        """
        self.writer.add_scalar(
            "{}/_loss/{}".format(val_or_train, info_for_writer), self.losses.get_avg(), epoch)
        self.writer.add_scalar(
            "{}/_top1/{}".format(val_or_train, info_for_writer), self.top1.get_avg(), epoch)
        self.writer.add_scalar(
            "{}/_loss/{}".format(val_or_train, info_for_writer), self.top5.get_avg(), epoch)

        self.logger.info(
            "{} : [{:3d}/{}] Final Loss {:.3f} Final Prec@(1, 5) ({:.1%}, {:.1%}) Time {:.2f}" .format(
                val_or_train,
                epoch + 1,
                self.epochs,
                self.losses.get_avg(),
                self.top1.get_avg(),
                self.top5.get_avg(),
                time.time() - start_time))

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
        """
        Logging training infomation at each print_freq iteration
        """
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))

        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)

        if (step > 1 and step % print_freq == 0) or step == len_loader - 1:
            self.logger.info(
                "{} : [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} Prec@(1, 5) ({:.1%}, {:.1%})" .format(
                    val_or_train,
                    epoch + 1,
                    self.epochs,
                    step,
                    len_loader - 1,
                    self.losses.get_avg(),
                    self.top1.get_avg(),
                    self.top5.get_avg()))

    def _reset_average_tracker(self):
        for tracker in [self.top1, self.top5, self.losses]:
            tracker.reset()
