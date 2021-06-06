import os
import time

import torch

from utils import AverageMeter, accuracy, save

class CFTrainingAgent:
    """The training agent to train the supernet and the searched architecture.
    """
    def train_loop(self, model, train_loader, val_loader, agent):
        training_step = getattr(self, f"_{agent.agent_state}_training_step")
        validate_step = getattr(self, f"{agent.agent_state}_validate_step")

        best_val_performance = -float("inf")
        for epoch in range(agent.start_epochs, agent.epochs):
            agent.logger.info(f"Start to train for epoch {epoch}")
            agent.logger.info(f"Learning Rate : {agent.optimizer.param_groups[0]['lr']:.8f}")

            training_step(
                model,
                train_loader,
                agent,
                epoch)
            val_performance = validate(
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
        return self._validate(model ,val_loader, agent, epoch)

    def _evaluate_training_step(self, model, train_loader, agent, epoch):
        self._training_step(model, train_loader, agent, epoch)

    def _evaluate_validate_step(self, model, val_loader, agent, epoch):
        return self._validate(model ,val_loader, agent, epoch)

    @staticmethod
    def searching_evaluate():
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        with torch.no_grad():
            for step, (X, y) in enumerate(val_loader):
                X, y = X.to(device, non_blocking=True), \
                       y.to(device, non_blocking=True)
                N = X.shape[0]
                outs = model(X)

                loss = criterion(outs, y)
                prec1, prec5 = accuracy(outs, y, topk=(1, 5))

                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)
        return top1.get_avg(), top5.get_avg(), losses.get_avg()


    def _training_step(self, model, train_loader, agent, epoch):
        top1 = AverageMeter()
        top5 = AverageMeter()
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

            prec1, prec5 = accuracy(outs, y, topk=(1, 5))

            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if (step > 1 and step % print_freq == 0) or (step == len(train_loader) - 1):
                agent.logger.info(f"Train : [{(epoch+1):3d}/{self.epochs}]"
                                 f"Step {step:3d}/{len(train_loader)-1:3d} Loss {losses.get_avg():.3f}"
                                 f"Prec@(1, 5) ({top1.get_avg():.1%}, {top5.get_avg():.1%})")

        agent.writer.add_scalar("Train/_loss/", losses.get_avg(), epoch)
        agent.writer.add_scalar("Train/_top1/", top1.get_avg(), epoch)
        agent.writer.add_scalar("Train/_top5/", top5.get_avg(), epoch)

        agent.logger.info(
            f"Train: [{epoch+1:3d}/{self.epochs}] Final Loss {losses.get_avg():.3f}" 
            f"Final Prec@(1, 5) ({top1.get_avg():.1%}, {top5.get_avg():.1%})"
            f"Time {time.time() - start_time:.2f}")


    def _validate(self, model, val_loader, agent):
        model.eval()
        start_time = time.time()

        top1_avg, top5_avg, losses_avg = self.searching_evaluate(model, val_loader, agent.device, agent.criterion)

        agent.writer.add_scalar("Valid/_loss/", losses_avg, epoch)
        agent.writer.add_scalar("Valid/_top1/", top1_avg, epoch)
        agent.writer.add_scalar("Valid/_top5/", top5_avg, epoch)

        agent.logger.info(
            f"Valid : [{epoch+1:3d}/{self.epochs}] Final Loss {losses_avg:.3f}" 
            f"Final Prec@(1, 5) ({top1_avg:.1%}, {top5_avg:.1%})"
            f"Time {time.time() - start_time:.2f}")

        return top1_avg



