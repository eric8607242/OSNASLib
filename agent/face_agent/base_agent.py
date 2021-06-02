import os
import time

import torch

from utils import AverageMeter, accuracy, save

from ..base_agent import MetaAgent

class FRMetaAgent(MetaAgent):
    """Face recognition meta agent
    """
    def __init__(self, config, title):
        super(FRMetaAgent, self).__init__(config, title)

    def train_loop(self, model,
                         train_loader,
                         val_loader):
        best_val_metric = 0.0
        for epoch in range(self.start_epochs, self.epochs):
            self.logger.info(f"Start to train for epoch {epoch}")
            self.logger.info(f"Learning Rate : {self.optimizer.param_groups[0]["lr"]:.8f}")

            self._training_step(
                model,
                train_loader,
                epoch)
            val_metric = self.validate(
                model,
                val_loader,
                epoch)

            if val_metric > best_val_metric:
                self.logger.info(f"Best validation metric : {val_metric}. Save model!")
                best_top1_acc = val_top1
                save(
                    model,
                    self.config["experiment_path"]["best_checkpoint_path"],
                    self.optimizer,
                    self.lr_scheduler,
                    epoch + 1)

            save(
                model,
                os.path.join(
                    self.config["experiment_path"]["checkpoint_root_path"],
                    f"{self.agent_state}_{epoch}.pth"),
                self.optimizer,
                self.lr_scheduler,
                epoch + 1)


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
                self.logger.info(f"Train : [{(epoch+1):3d}/{self.epochs}]"
                                 f"Step {step:3d}/{len(train_loader)-1:3d} Loss {losses.get_avg():.3f}")

        self.writer.add_scalar("Train/_loss/", losses.get_avg(), epoch)

        self.logger.info(
            f"Train: [{epoch+1:3d}/{self.epochs}] Final Loss {losses.get_avg():.3f}" 
            f"Time {time.time() - start_time:.2f}")

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
