import os
import time

from ..utils import *


class Trainer:
    def __init__(self, criterion, optimizer, lr_scheduler, writer, logger, device, trainer_state, args, training_strategy=None, search_strategy=None, start_epoch=0):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()
        self.device = device

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.trainer_state = trainer_state
        if self.trainer_state == "search":
            self.training_strategy = training_strategy
            self.search_strategy = search_strategy

        self.writer = writer
        self.logger = logger

        self.start_epoch = start_epoch
        self.epochs = args.epochs

        self.args = args


    def train_loop(self, model, train_loader, val_loader):
        """
        Support mode:
        1) Two stages training:
            - Uniform sampling
            - Strict sampling
        2) Jointly training:
            - Training architecture parameter and supernet
        """
        best_top1_acc = 0.0
        

        for epoch in range(self.start_epoch, self.epochs):
            self.logger.info("Start to train for epoch {}".format(epoch))
            self.logger.info("Learning Rate : {:.8f}".format(self.optimizer.param_groups[0]["lr"]))

            self._training_step(model, train_loader, epoch)
            val_top1 = self.validate(model, val_loader, epoch, inference=False if self.trainer_state=="search" else True)

            if val_top1 > best_top1_acc:
                self.logger.info("Best validation top1-acc : {}. Save model!".format(val_top1*100))
                best_top1_acc = val_top1
                save(model, self.args.best_model_path, self.optimizer, self.lr_scheduler, epoch+1)

            save(model, os.path.join(self.args.checkpoint_path_root, "{}_{}.pth".format(self.trainer_state, epoch)), self.optimizer, self.lr_scheduler, epoch+1)
                

    def _training_step(self, model, train_loader, epoch):
        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(train_loader):

            if self.trainer_state == "search":
                # If search strategy is "differentiable". Update architecture parameter.
                self.search_strategy.step()

            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            self.optimizer.zero_grad()

            if self.trainer_state == "search":
                self.training_strategy.step()

            outs = model(X)

            loss = self.criterion(outs, y)
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(train_loader), val_or_train="Train")

        self._epoch_stats_logging(start_time, epoch, val_or_train="Train")
        self._reset_average_tracker()


    def validate(self, model, val_loader, epoch, inference=True):
        """
            inference(bool) : True, evaluate specific architecture.
                              False, evaluate supernet.
        """
        model.eval()
        start_time = time.time()

        for step, (X, y) in enumerate(val_loader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            if not inference:
                self.training_strategy.step()

            outs = model(X)

            loss = self.criterion(outs, y)
            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(val_loader), val_or_train="Valid")

        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time, epoch, val_or_train="Valid")
        self._reset_average_tracker()

        return top1_avg


    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_writer=""):
        """
        Logging training information and record information in Tensorboard for each epoch
        """
        self.writer.add_scalar("{}/_loss/{}".format(val_or_train, info_for_writer), self.losses.get_avg(), epoch)
        self.writer.add_scalar("{}/_top1/{}".format(val_or_train, info_for_writer), self.top1.get_avg(), epoch)
        self.writer.add_scalar("{}/_loss/{}".format(val_or_train, info_for_writer), self.top5.get_avg(), epoch)

        self.logger.info("{} : [{:3d}/{}] Final Loss {:.3f} Final Prec@(1, 5) ({:.1%}, {:.1%}) Time {:.2f}"\
                    .format(val_or_train, epoch+1, self.epochs, self.losses.get_avg(), self.top1.get_avg(), self.top5.get_avg(), time.time()-start_time))


    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train, print_freq=100):
        """
        Logging training infomation at each print_freq iteration
        """
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))

        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)

        if (step > 1 and step % print_freq == 0) or step == len_loader-1:
            self.logger.info("{} : [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} Prec@(1, 5) ({:.1%}, {:.1%})"\
                    .format(val_or_train, epoch+1, self.epochs, step, len_loader-1, self.losses.get_avg(), self.top1.get_avg(), self.top5.get_avg()))


    def _reset_average_tracker(self):
        for tracker in [self.top1, self.top5, self.losses]:
            tracker.reset()

