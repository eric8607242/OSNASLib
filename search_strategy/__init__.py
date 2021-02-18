from .random_search import random_search
from .evolution_search import evoluation_algorithm

from ..lib import *


class SearchStrategy:
    def __init__(self, supernet, val_loader, search_strategy, args, logger):
        self.search_strategy = search_strategy

        if self.search_strategy == "differentaible":
            # Init architecture parameter optimizer
            self.a_optimizer = get_optimizer(supernet.get_architecture_param(),
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

        self.ce_criterion = None

        self.val_loader = val_loader

    def step(self):
        if self.search_strategy == "evolution":
            pass

        elif self.search_strategy == "random_search":
            pass

        elif self.search_strategy == "differentaible":
            # Update architecture parameter
            self.a_optimizer.zero_grad()

            X, y = next(iter(self.val_loader))
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            outs = self.supernet(X)
            ce_loss = self.ce_criterion(outs, y)

            architecture_parameter = self.supernet.get_architecture_param()
            architecture_info = lookup_table.get_model_info(architecture_parameter, info_metric=self.args.info_metric)

            hc_loss = self.hc_criterion(args.target_hc, architecture_info)

            total_loss = hc_loss + ce_loss
            total_loss.backward()

            self.a_optimizer.step()

        else:
            raise 

    def search(self, trainer, training_strategy, lookup_table):
        if self.search_strategy == "evolution":
            best_architecture = evoluation_algorithm(trainer, training_strategy, self.supernet, self.val_loader, lookup_table, self.args.target_hc, self.logger, generation_num=self.args.generation_num, population=self.args.population, parent_num=self.args.parent_num, info_metric=self.args.info_metric)

        elif self.search_strategy == "random_search":
            best_architecture = random_search(trainer, training_strategy, self.supernet, self.val_loader, lookup_table, self.args.target_hc, self.logger, random_iteration=self.args.random_iteration, info_metric=self.args.info_metric)

        elif self.search_strategy == "differentaible":
            best_architecture = self.supernet.get_best_architecture()

            self.supernet.module.set_activate_architecture(best_architecture) if isinstance(self.supernet, nn.DataParallel) else self.supernet.set_activate_architecture(best_architecture)
            best_architecture_top1 = trainer.validate(self.supernet, self.val_loader)
        else:
            raise 

        return best_architecture
