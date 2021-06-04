import os
import time

import torch

from utils import AverageMeter, accuracy, save

from ..base_agent import MetaAgent

class FRMetaAgent(MetaAgent):
    """Face recognition meta agent
    """
    evaluate_metric = "acc"
    def _training_step(
            self,
            model,
            train_loader,
            epoch,
            print_freq=100):
        losses = AverageMeter()

        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(train_loader):
            self._iteration_preprocess()

            X, y = X.to(
                self.device, non_blocking=True), y.to(
                self.device, non_blocking=True)
            N = X.shape[0]

            self.optimizer.zero_grad()
            outs = model(X)

            loss = self.criterion(outs, y)
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            losses.update(loss.item(), N)

            if (step > 1 and step % print_freq == 0) or (step == len(train_loader) - 1):
                self.logger.info(f"Train : [{(epoch+1):3d}/{self.epochs}] "
                                 f"Step {step:3d}/{len(train_loader)-1:3d} Loss {losses.get_avg():.3f} ")

        self.writer.add_scalar("Train/_loss/", losses.get_avg(), epoch)

        self.logger.info(
            f"Train: [{epoch+1:3d}/{self.epochs}] Final Loss {losses.get_avg():.3f} " 
            f"Time {time.time() - start_time:.2f} ")

    @staticmethod
    def searching_evaluate(model, val_loader, device, criterion):
        losses = AverageMeter()
        with torch.no_grad():
            for step, (X, y) in enumerate(val_loader):
                X, y = X.to(device, non_blocking=True), \
                       y.to(device, non_blocking=True)
                N = X.shape[0]
                outs = model(X)

                loss = criterion(outs, y)
                losses.update(loss.item(), N)

        # Make search strategy cam compare the architecture performance
        return -losses.get_avg(),
