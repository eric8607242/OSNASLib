import torch.nn as nn

from .random_search import random_search
from .evolution_search import evoluation_algorithm

class SearchStrategy:
    def __init__(
            self,
            supernet,
            val_loader,
            lookup_table,
            search_strategy,
            args,
            logger,
            device):
        self.search_strategy = search_strategy

        if self.search_strategy == "differentiable" or self.search_strategy == "differentiable_gumbel":
            # Init architecture parameter optimizer
            self.a_optimizer = get_optimizer(
                [
                    supernet.module.get_architecture_param() if isinstance(
                        supernet,
                        nn.DataParallel) else supernet.get_architecture_param()],
                args.a_optimizer,
                learning_rate=args.a_lr,
                weight_decay=args.a_weight_decay,
                logger=logger,
                momentum=args.a_momentum,
                alpha=args.a_alpha,
                beta=args.a_beta)
            self.hc_criterion = get_hc_criterion()

        self.args = args
        self.logger = logger
        self.supernet = supernet

        self.ce_criterion = get_criterion()
        self.val_loader = val_loader

        self.lookup_table = lookup_table

        self.device = device

    def step(self):
        if self.search_strategy == "evolution":
            pass

        elif self.search_strategy == "random_search":
            pass

        elif self.search_strategy == "differentiable" or self.search_strategy == "differentiable_gumbel":
            self.supernet.module.set_forward_state("sum") if isinstance(
                self.supernet, nn.DataParallel) else self.supernet.set_forward_state("sum")

            # Update architecture parameter
            self.a_optimizer.zero_grad()

            X, y = next(iter(self.val_loader))
            X, y = X.to(
                self.device, non_blocking=True), y.to(
                self.device, non_blocking=True)

            outs = self.supernet(X)
            ce_loss = self.ce_criterion(outs, y)

            architecture_parameter = self.supernet.module.get_architecture_param() if isinstance(
                self.supernet, nn.DataParallel) else self.supernet.get_architecture_param()
            architecture_info = self.lookup_table.get_model_info(
                architecture_parameter, info_metric=self.args.info_metric)

            hc_loss = self.hc_criterion(
                self.args.target_hc,
                architecture_info,
                self.args.hc_weight)

            total_loss = hc_loss + ce_loss
            total_loss.backward()

            self.a_optimizer.step()
            self.a_optimizer.zero_grad()

        else:
            raise

    def search(self, trainer, training_strategy):
        if self.search_strategy == "evolution":
            best_architecture = evoluation_algorithm(
                trainer,
                training_strategy,
                self.supernet,
                self.val_loader,
                self.lookup_table,
                self.args.target_hc,
                self.logger,
                generation_num=self.args.generation_num,
                population=self.args.population,
                parent_num=self.args.parent_num,
                info_metric=self.args.info_metric)

        elif self.search_strategy == "random_search":
            best_architecture = random_search(
                trainer,
                training_strategy,
                self.supernet,
                self.val_loader,
                self.lookup_table,
                self.args.target_hc,
                self.logger,
                random_iteration=self.args.random_iteration,
                info_metric=self.args.info_metric)

        elif self.search_strategy == "differentiable" or self.search_strategy == "differentiable_gumbel":
            best_architecture = self.supernet.module.get_best_architecture_param() if isinstance(
                self.supernet, nn.DataParallel) else self.supernet.get_best_architecture_param()

            self.supernet.module.set_activate_architecture(best_architecture) if isinstance(
                self.supernet,
                nn.DataParallel) else self.supernet.set_activate_architecture(best_architecture)
            best_architecture_top1 = trainer.validate(
                self.supernet, self.val_loader, 0)
        else:
            raise

        best_architecture_hc = self.lookup_table.get_model_info(
            best_architecture)

        return best_architecture, best_architecture_hc, best_architecture_top1
