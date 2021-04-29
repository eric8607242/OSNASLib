from utils import get_optimizer
from criterion import get_hc_criterion

from .base import BaseSearcher

class DifferentiableSearcher(BaseSearcher):
    def __init__(self, supernet, val_loader, lookup_table, training_strategy, device, logger):
        super(DifferentiableSearcher, self).__init__(supernet, val_loader, lookup_table, training_strategy, device, logger)

        # Init architecture parameter optimizer
        self.a_optimizer = get_optimizer(
            [supernet.module.get_architecture_param() if isinstance(supernet, nn.DataParallel) \
                    else supernet.get_architecture_param()],
            config["arch_optim"]["a_optimizer"],
            learning_rate=config["arch_optim"]["a_lr"],
            weight_decay=config["arch_optim"]["a_weight_decay"],
            logger=logger,
            momentum=config["arch_optim"]["a_momentum"],
            alpha=config["arch_optim"]["a_alpha"],
            beta=config["arch_optim"]["a_beta"])

        self.hc_criterion = get_hc_criterion()
    

    def step(self):
        self.supernet.module.set_forward_state("sum") if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.set_forward_state("sum")

        # Update architecture parameter
        self.a_optimizer.zero_grad()

        X, y = next(iter(self.val_loader))
        X, y = X.to(
            self.device, non_blocking=True), y.to(
            self.device, non_blocking=True)

        outs = self.supernet(X)
        ce_loss = self.criterion(outs, y)

        architecture_parameter = self.supernet.module.get_architecture_param() if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.get_architecture_param()
        architecture_info = self.lookup_table.get_model_info(
            architecture_parameter, info_metric=self.config.info_metric)

        hc_loss = self.hc_criterion(
            self.target_hc,
            architecture_info,
            self.config["arch_optim"]["hc_weight"])

        total_loss = hc_loss + ce_loss
        total_loss.backward()

        self.a_optimizer.step()
        self.a_optimizer.zero_grad()


    def search(self):
        best_architecture = self.supernet.module.get_best_architecture_param() if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.get_best_architecture_param()

        #self.supernet.module.set_activate_architecture(best_architecture) if isinstance(
            #self.supernet,
            #nn.DataParallel) else self.supernet.set_activate_architecture(best_architecture)
        #best_architecture_top1 = trainer.validate(
            #self.supernet, self.val_loader, 0)
        return best_architecture
