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

        self.hc_criterion = get_hc_criterion(self.config["train"]["hc_criterion_type"])

        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()
        self.hc_losses = AverageMeter()
        self.ce_losses = AverageMeter()
    
        self.step_num = 0

    def step(self):
        self.step_num += 1
        
        self.supernet.module.set_forward_state("sum") if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.set_forward_state("sum")

        # Update architecture parameter
        self.a_optimizer.zero_grad()

        architecture_parameter = self.supernet.module.get_architecture_param() if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.get_architecture_param()
        architecture_info = self.lookup_table.get_model_info(
            architecture_parameter)

        hc_loss = self.hc_criterion(architecture_info, self.target_hc)*self.config["arch_optim"]["hc_weight"]

        X, y = next(iter(self.val_loader))
        X, y = X.to(
            self.device, non_blocking=True), y.to(
            self.device, non_blocking=True)

        outs = self.supernet(X)
        ce_loss = self.criterion(outs, y)

        total_loss = hc_loss + ce_loss
        self.logger.info(f"Hardware loss : {hc_loss.item()}")

        total_loss.backward()
        self.a_optimizer.step()

        self._intermediate_stats_logging(
            outs,
            y,
            ce_loss,
            hc_loss,
            N,
            len_loader=len(self.val_loader),
            val_or_train="ArchParam_Train")

    def search(self):
        best_architecture = self.supernet.module.get_best_architecture_param() if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.get_best_architecture_param()

        #self.supernet.module.set_activate_architecture(best_architecture) if isinstance(
            #self.supernet,
            #nn.DataParallel) else self.supernet.set_activate_architecture(best_architecture)
        #best_architecture_top1 = trainer.validate(
            #self.supernet, self.val_loader, 0)
        return best_architecture
    

    def _intermediate_stats_logging(self, outs, y, ce_loss, hc_loss, N, len_loader, val_or_train, print_freq=100):
        step = self.step_num

        total_loss = ce_loss + hc_loss
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))

        self.losses.update(total_loss.item(), N)
        self.hc_losses.update(hc_loss.item(), N)
        self.ce_losses.update(ce_loss.item(), N)

        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)

        if (step > 1 and step % print_freq == 0) or step == len_loader - 1:
            self.logger.info("{} : Step {:03d}/{:03d} Loss {:.3f} CE Loss {:.3f} HC Loss {:.3f} Prec@(1, 5) ({:.1%}, {:.1%})" .format(
                    val_or_train,
                    step,
                    len_loader - 1,
                    self.losses.get_avg(),
                    self.ce_losses.get_avg(),
                    self.hc_losses.get_avg(),
                    self.top1.get_avg(),
                    self.top5.get_avg()))

            if step == len_loader - 1:
                self.step_num = 0


    def _reset_average_tracker(self):
        for tracker in [self.top1, self.top5, self.losses, self.ce_losses, self.hc_losses]:
            tracker.reset()
