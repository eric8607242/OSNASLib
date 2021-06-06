import os
import time

import torch

from utils import AverageMeter, save

class {{customize_class}}TrainingAgent:
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
        pass

    def _search_validate_step(self, model, val_loader, agent, epoch):
        return evaluate_metric

    def _evaluate_training_step(self, model, train_loader, agent, epoch):
        pass

    def _evaluate_validate_step(self, model, val_loader, agent, epoch):
        return evaluate_metric

    @staticmethod
    def searching_evaluate(model, val_loader, device, criterion):
        return evaluate_metric


