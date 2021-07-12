import os

from abc import ABC, abstractmethod

from utils import save

class MetaTrainingAgent:
    def train_loop(self, model, train_loader, val_loader, agent):
        """ The main training loop.

        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            val_loader (torch.utils.data.DataLoader)
            agent (Object)
        """
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
                    agent.criterion,
                    agent.optimizer,
                    agent.lr_scheduler,
                    epoch + 1)

            save(
                model,
                os.path.join(
                    agent.config["experiment_path"]["checkpoint_root_path"],
                    f"{agent.agent_state}_{epoch}.pth"),
                agent.criterion,
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
        self._training_step(model, train_loader, agent, epoch)

    def _evaluate_training_step(self, model, train_loader, agent, epoch):
        """ The training step for evaluating process (training from scratch).

        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)
        """
        self._training_step(model, train_loader, agent, epoch)

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
        return self._validate_step(model, val_loader, agent, epoch)

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
        return self._validate_step(model, val_loader, agent, epoch)

    @abstractmethod
    def _training_step(self, model, train_loader, agent, epoch):
        """
        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)
        """
        raise NotImplementedError

    @abstractmethod
    def _validate_step(self, model, val_loader, agent, epoch):
        """
        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)

        Return:
            evaluate_metric (float): The performance of the searched model.
        """
        raise NotImplementedError
        return evaluate_metric
