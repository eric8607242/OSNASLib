import os
import time

import numpy as np

import torch

from utils import AverageMeter, save

from .face_evaluate import evaluate

class FRTrainingAgent:
    """The training agent to train the supernet and the searched architecture.
    """
    def train_loop(self, model, train_loader, val_loader, agent):
        training_step = getattr(self, f"_{agent.agent_state}_training_step")
        validate_step = getattr(self, f"_{agent.agent_state}_validate_step")

        best_val_performance = -float("inf")
        for epoch in range(agent.start_epochs, agent.epochs):
            agent.logger.info(f"Start to train for epoch {epoch}")
            agent.logger.info(f"Learning Rate : {agent.optimizer.param_groups[0]['lr']:.8f}")

            training_step(
                model,
                train_loader,
                agent,
                epoch)
            val_performance = validate_step(
                model,
                val_loader,
                agent,
                epoch)

            if val_performance > best_val_performance:
                agent.logger.info(f"Best validation performance : {val_performance}. Save model!")
                best_val_performance = val_performance
                save(
                    model,
                    agent.config["experiment_path"]["best_checkpoint_path"],
                    agent.optimizer,
                    agent.lr_scheduler,
                    epoch + 1)

            save(
                model,
                os.path.join(
                    agent.config["experiment_path"]["checkpoint_root_path"],
                    f"{agent.agent_state}_{epoch}.pth"),
                agent.optimizer,
                agent.lr_scheduler,
                epoch + 1)


    def _search_training_step(self, model, train_loader, agent, epoch):
        self._training_step(model, train_loader, agent, epoch)

    def _search_validate_step(self, model, val_loader, agent, epoch):
        model.eval()
        start_time = time.time()

        agent._iteration_preprocess()
        minus_losses_avg = self.searching_evaluate(model, val_loader, agent.device, agent.criterion)[0]

        agent.writer.add_scalar("Valid/_losses/", -minus_losses_avg, epoch)

        agent.logger.info(
            f"Valid : [{epoch+1:3d}/{agent.epochs}]" 
            f"Final Losses: {-minus_losses_avg:.2f}"
            f"Time {time.time() - start_time:.2f}")

        return minus_losses_avg

    def _evaluate_training_step(self, model, train_loader, agent, epoch):
        self._training_step(model, train_loader, agent, epoch)

    def _evaluate_validate_step(self, model, val_loader, agent, epoch):
        model.eval()
        start_time = time.time()

        all_labels = []
        all_embeds1, all_embeds2 = [], []

        with torch.no_grad():
            for idx, ((imgs1, imgs2), labels) in enumerate(val_loader):
                # Move data sample
                batch_size = labels.size(0)
                imgs1 = imgs1.to(agent.device)
                imgs2 = imgs2.to(agent.device)
                labels = labels.to(agent.device)
                # Extract embeddings
                embeds1 = model(imgs1)
                embeds2 = model(imgs2)
                # Accumulates
                all_labels.append(labels.detach().cpu().numpy())
                all_embeds1.append(embeds1.detach().cpu().numpy())
                all_embeds2.append(embeds2.detach().cpu().numpy())

        # Evaluate
        labels = np.concatenate(all_labels)
        embeds1 = np.concatenate(all_embeds1)
        embeds2 = np.concatenate(all_embeds2)
        TP_ratio, FP_ratio, accs, best_thresholds = evaluate(embeds1, embeds2, labels)
        # Save Checkpoint
        acc_avg = accs.mean()
        thresh_avg = best_thresholds.mean()

        agent.writer.add_scalar("Valid/_acc/", acc_avg, epoch)
        agent.writer.add_scalar("Valid/_thresh/", thresh_avg, epoch)

        agent.logger.info(
            f"Valid : [{epoch+1:3d}/{agent.epochs}] " 
            f"Final Acc : {acc_avg:.2f} Final Thresh : {thresh_avg:.2f} "
            f"Time {time.time() - start_time:.2f}")

        return acc_avg

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

    def _training_step(
            self,
            model,
            train_loader,
            agent,
            epoch,
            print_freq=100):
        losses = AverageMeter()

        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(train_loader):
            if agent.agent_state == "search":
                agent._iteration_preprocess()

            X, y = X.to(
                agent.device, non_blocking=True), y.to(
                agent.device, non_blocking=True)
            N = X.shape[0]

            agent.optimizer.zero_grad()
            outs = model(X)

            loss = agent.criterion(outs, y)
            loss.backward()

            agent.optimizer.step()
            agent.lr_scheduler.step()

            losses.update(loss.item(), N)

            if (step > 1 and step % print_freq == 0) or (step == len(train_loader) - 1):
                agent.logger.info(f"Train : [{(epoch+1):3d}/{agent.epochs}] "
                                 f"Step {step:3d}/{len(train_loader)-1:3d} Loss {losses.get_avg():.3f} ")

        agent.writer.add_scalar("Train/_loss/", losses.get_avg(), epoch)

        agent.logger.info(
            f"Train: [{epoch+1:3d}/{agent.epochs}] Final Loss {losses.get_avg():.3f} " 
            f"Time {time.time() - start_time:.2f} ")



