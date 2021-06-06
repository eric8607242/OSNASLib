import torch
import torch.nn as nn

from utils import get_optimizer, min_max_normalize, save, AverageMeter, accuracy
from criterion import get_hc_criterion, get_criterion

from .base import BaseSearcher

class DifferentiableSearcher(BaseSearcher):
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger):
        super(DifferentiableSearcher, self).__init__(config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger)

        # Init architecture parameters
        self.supernet.initialize_arch_param()

        # Init architecture parameters optimizer
        self.a_optimizer = get_optimizer(
            [self.supernet.module.get_arch_param() if isinstance(self.supernet, nn.DataParallel) \
                    else self.supernet.get_arch_param()],
            self.config["arch_optim"]["a_optimizer"],
            learning_rate=self.config["arch_optim"]["a_lr"],
            weight_decay=self.config["arch_optim"]["a_weight_decay"],
            logger=self.logger,
            momentum=self.config["arch_optim"]["a_momentum"],
            alpha=self.config["arch_optim"]["a_alpha"],
            beta=self.config["arch_optim"]["a_beta"])

        self.criterion = get_criterion(self.config["agent"]["criterion_agent"])
        self.hc_criterion = get_hc_criterion(self.config["agent"]["hc_criterion_agent"], self.config["criterion"])

        self.step_num = 0


    def step(self, print_freq=100):
        losses = AverageMeter()
        hc_losses = AverageMeter()
        ce_losses = AverageMeter()

        self.step_num += 1
        
        self.supernet.module.set_forward_state("sum") if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.set_forward_state("sum")

        # Update architecture parameter
        self.a_optimizer.zero_grad()

        architecture_parameter = self.supernet.module.get_arch_param() if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.get_arch_param()
        architecture_info = self.lookup_table.get_model_info(
            architecture_parameter)

        target_hc_tensor = torch.tensor(self.target_hc, dtype=torch.float).to(self.device)
        hc_loss = self.hc_criterion(architecture_info, target_hc_tensor)*self.config["criterion"]["hc_weight"]

        X, y = next(iter(self.val_loader))
        X, y = X.to(
            self.device, non_blocking=True), y.to(
            self.device, non_blocking=True)
        N = X.shape[0]

        outs = self.supernet(X)
        ce_loss = self.criterion(outs, y)

        total_loss = hc_loss + ce_loss

        total_loss.backward()
        self.a_optimizer.step()

        losses.update(total_loss.item(), N)
        hc_losses.update(hc_loss.item(), N)
        ce_losses.update(ce_loss.item(), N)

        if (self.step_num > 1 and self.step_num % print_freq == 0) or self.step_num == len(self.val_loader)- 1:
            self.logger.info(f"ArchParam Train : Step {self.step_num:03d}/{len(self.val_loader)-1:03d} "
                             f"Loss {losses.get_avg():.3f} CE Loss {ce_losses.get_avg():.3f} HC Loss {hc_losses.get_avg():.3f}")

        if self.step_num == len(self.val_loader) - 1:
            self.step_num = 0

        
    def search(self):
        best_architecture = self.supernet.module.get_best_arch_param() if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.get_best_arch_param()

        return best_architecture, _, _
