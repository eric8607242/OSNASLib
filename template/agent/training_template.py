import os
import time

import torch

from utils import AverageMeter, save

class {{customize_class}}TrainingAgent:
    """The training agent to train the supernet and the searched architecture.

    By implementing TrainingAgent class, users can adapt the searching and evaluating agent into
    various tasks easily.
    """
    def train_loop(self, model, train_loader, val_loader, agent):
        """ The main training loop.

        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            val_loader (torch.utils.data.DataLoader)
            agent (Object)
        """
        # Utilize different step method based on differet agent state
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
        """ The training step for searching process. Users should step the sampler
        to decide how to train supernet and step the search strategy to search the architecture.

        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            agent (Object): The search agent.
            epoch (int)
        """
        model.train()
        for step, (X, y) in enumerate(train_loader):
            agent._iteration_preprocess()
            raise NotImplemented

    def _search_validate_step(self, model, val_loader, agent, epoch):
        """ The validate step for searching process.

        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            agent (Object): The search agent.
            epoch (int)

        Return:
            evaluate_metric (float): The performance of the supernet
        """
        model.eval()
        with torch.no_grad():
            for step, (X, y) in enumerate(val_loader):
                agent._iteration_preprocess()

                raise NotImplemented
        return evaluate_metric

    def _evaluate_training_step(self, model, train_loader, agent, epoch):
        """ The training step for evaluating process (training from scratch).

        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)
        """
        model.train()
        for step, (X, y) in enumerate(train_loader):
            raise NotImplemented

    def _evaluate_validate_step(self, model, val_loader, agent, epoch):
        """ The training step for evaluating process (training from scratch).

        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)

        Return:
            evaluate_metric (float): The performance of the searched model.
        """
        model.eval()
        with torch.no_grad():
            for step, (X, y) in enumerate(val_loader):
                raise NotImplemented
        return evaluate_metric

    @staticmethod
    def searching_evaluate(model, val_loader, device, criterion):
        """ Evaluating the performance of the supernet. The search strategy will evaluate
        the architectures by this static method to search.

        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            device (torch.device)
            criterion (nn.Module)

        Return:
            evaluate_metric (float): The performance of the supernet.
        """
        model.eval()
        with torch.no_grad():
            for step, (X, y) in enumerate(val_loader):
                raise NotImplemented
        return evaluate_metric


