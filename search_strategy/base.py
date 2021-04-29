import torch
import torch.nn as nn

from utils import AverageMeter, accuracy

class BaseSearcher:
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, logger):
        self.config = config
        self.logger = logger

        self.supernet = supernet
        self.val_loader = val_loader

        self.lookup_table = lookup_table
        self.training_strategy = training_strategy

        self.device = device
        self.logger = logger

        self.target_hc = self.config["target_hc"]
        self.info_metric = self.config["search_utility"]["info_metric"]
        self.top1 = AverageMeter()

    def step(self):
        pass

    def _evaluate(self):
        self.supernet.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(self.val_loader):
                X, y = X.to(
                    self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True)
                N = X.shape[0]

                outs = self.supernet(X)
                prec1, prec5 = accuracy(outs, y, topk=(1, 5))
                self.top1.update(prec1.item(), N)

        top1_avg = self.top1.get_avg()
        self.top1.reset()
        return top1_avg

    def evaluate_architectures(self, architectures_list):
        architectures_top1_acc = []
        for i, architecture in enumerate(architectures_list):
            self.supernet.module.set_activate_architecture(architecture) if isinstance(
                self.supernet, nn.DataParallel) else self.supernet.set_activate_architecture(architecture)

            top1_avg = self._evaluate()
            architectures_top1_acc.append(top1_avg)
            self.logger.info(f"Evaluate {i} architecture top1-avg : {100*top1_avg}%")

        architectures_top1_acc = np.array(architectures_top1_acc)
        return architectures_top1_acc

