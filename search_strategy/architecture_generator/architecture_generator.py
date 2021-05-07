import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import scipy.stats as stats

from utils import get_optimizer, min_max_normalize, save, AverageMeter, accuracy
from criterion import get_hc_criterion, get_criterion

from ..base import BaseSearcher
from .generator import Generator
from .prior_pool import PriorPool

class ArchitectureGeneratorSearcher(BaseSearcher):
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, logger):
        super(ArchitectureGeneratorSearcher, self).__init__(config, supernet, val_loader, lookup_table, training_strategy, device, logger)

        generator = Generator(self.config["generator"]["hc_dim"])
        self.generator = generator.to(self.device)


        self.g_optimizer = get_optimizer(self.generator.parameters(),
                        self.config["arch_optim"]["a_optimizer"],
                        learning_rate=self.config["arch_optim"]["a_lr"],
                        weight_decay=self.config["arch_optim"]["a_weight_decay"],
                        logger=self.logger,
                        momentum=self.config["arch_optim"]["a_momentum"],
                        alpha=self.config["arch_optim"]["a_alpha"],
                        beta=self.config["arch_optim"]["a_beta"])

        self.criterion = get_criterion(self.config["train"]["criterion_type"])
        self.hc_criterion = get_hc_criterion(self.config["arch_optim"]["hc_weight"])

        self.arch_param_nums = self.supernet.get_arch_param_nums()
        self.prior_pool = PriorPool(self.lookup_table, self.arch_param_nums, self.config, self.logger)

        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()
        self.hc_losses = AverageMeter()
        self.ce_losses = AverageMeter()

        self.hardware_constraint_pool = [i for i in range(self.config["search_utility"]["lowest_hardware_constraint"], self.config["search_utility"]["highest_hardware_constraint"], 5)]
        self.hardware_constraint_index = 0
        random.shuffle(self.hardware_constraint_pool)
        
        self.supernet.module.set_forward_state("sum") if isinstance(
            self.supernet, nn.DataParallel) else self.supernet.set_forward_state("sum")

        self.search_epochs = self.config["generator"]["epochs"]


    def search(self):
        tau = 5
        
        best_loss = float("inf")
        best_top1 = 0
        for epoch in range(self.search_epochs):
            self.logger.info(f"Start to train the architecture generator for epoch {epoch}")
            self.logger.info(f"Tau : {tau}")

            self._generator_training_step(tau, epoch=epoch)
            target_val_hardware_constraint = (self.config["search_utility"]["lowest_hardware_constraint"] + self.config["search_utility"]["highest_hardware_constraint"]) / 2
            top1_avg = self._generator_validate(target_hardware_constraint=target_val_hardware_constraint, epoch=epoch)

            evaluate_metric, total_loss, kendall_tau = self._evaluate_generator()
            self.logger.info(f"The evaluating total loss : {total_loss}")
            self.logger.info(f"The Kendall Tau : {kendall_tau}")

            if total_loss < best_loss:
                self.logger.info(f"The newest best loss achieve {total_loss}! Save model.")
                save(self.generator, self.config["experiment_path"]["best_loss_generator_checkpoint_path"], self.g_optimizer, None, epoch+1)
                best_loss = total_loss

            if best_top1 < top1_avg:
                self.logger.info(f"The newest best top1 accuracy achieve {top1_avg}! Save model.")
                save(self.generator, self.config["experiment_path"]["best_acc_generator_checkpoint_path"], self.g_optimizer, None, epoch+1)
                best_top1 = top1_avg

            tau *= self.config["generator"]["tau_decay"]


    def _generator_training_step(self, tau, epoch):
        for step, (X, y) in enumerate(self.val_loader):
            self.g_optimizer.zero_grad() 
            target_hardware_constraint = self._get_target_hardware_constraint()

            arch_param = self._get_arch_param(target_hardware_constraint)
            arch_param = self._set_arch_param(arch_param, tau)

            arch_param_hardware_constraint = self.lookup_table.get_model_info(arch_param)
            self.logger.info(f"Generating architecture parameter hardware constraint: {arch_param_hardware_constraint.item()}")

            hc_loss = self.hc_criterion(target_hardware_constraint, arch_param_hardware_constraint)

            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            outs = self.supernet(X)

            ce_loss = self.criterion(outs, y)
            total_loss = ce_loss + hc_loss
            self.logger.info(f"Hardware loss : {hc_loss.item()}")

            total_loss.backward()
            self.g_optimizer.step()

            self._intermediate_stats_logging(
                outs,
                y,
                ce_loss,
                hc_loss,
                step,
                epoch,
                N,
                len_loader=len(self.val_loader),
                val_or_train="Generator_Train")
        self._reset_average_tracker()


    def _generator_validate(self, target_hardware_constraint, epoch):
        with torch.no_grad():
            target_hardware_constraint = self._get_target_hardware_constraint(target_hardware_constraint)
            arch_param = self._get_arch_param(target_hardware_constraint)
            arch_param = self._set_arch_param(arch_param)

            arch_param_hardware_constraint = self.lookup_table.get_model_info(arch_param)
            self.logger.info(f"Generating architecture parameter hardware constraint: {arch_param_hardware_constraint.item()}")

            hc_loss = self.hc_criterion(target_hardware_constraint, arch_param_hardware_constraint)
            self.logger.info(f"Hardware loss : {hc_loss.item()}")

            for step, (X, y) in enumerate(self.val_loader):
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                N = X.shape[0]

                outs = self.supernet(X)
                ce_loss = self.criterion(outs, y)

                self._intermediate_stats_logging(
                    outs,
                    y,
                    ce_loss,
                    hc_loss,
                    step,
                    epoch,
                    N,
                    len_loader=len(self.val_loader),
                    val_or_train="Generator_Valid")
        top1_avg = self.top1.get_avg()
        self._reset_average_tracker()

        return top1_avg


    def _get_target_hardware_constraint(self, target_hardware_constraint=None):
        if target_hardware_constraint is None:
            target_hardware_constraint = self.hardware_constraint_pool[self.hardware_constraint_index] + random.random() - 0.5

            self.hardware_constraint_index += 1
            if self.hardware_constraint_index == len(self.hardware_constraint_pool):
                self.hardware_constraint_index = 0
                random.shuffle(self.hardware_constraint_pool)

        target_hardware_constraint = torch.tensor(target_hardware_constraint, dtype=torch.float32).view(-1, 1)
        target_hardware_constraint = target_hardware_constraint.to(self.device)

        return target_hardware_constraint


    def _set_arch_param(self, arch_param, tau=None):
        if tau is not None:
            arch_param = self.prior_pool.get_probability_arch_param(arch_param, tau)
        else:
            arch_param = self.prior_pool.get_validation_arch_param(arch_param)

        arch_param = arch_param.to(self.device)
        self.supernet.module.set_arch_param(arch_param) if isinstance(self.supernet, nn.DataParallel) else self.supernet.set_arch_param(arch_param)
        return arch_param


    def _get_arch_param(self, target_hardware_constraint):
        hardware_constraint = target_hardware_constraint.to(self.device)
        self.logger.info(f"Target hardware constraint: {hardware_constraint.item()}")

        prior = self.prior_pool.get_prior(hardware_constraint.item())
        prior = prior.to(self.device)

        normalize_hardware_constraint = min_max_normalize(self.config["search_utility"]["lowest_hardware_constraint"], self.config["search_utility"]["highest_hardware_constraint"], hardware_constraint)

        arch_param = self.generator(prior, normalize_hardware_constraint)
        arch_param = arch_param.reshape(self.arch_param_nums)
        return arch_param

    
    def _intermediate_stats_logging(self, outs, y, ce_loss, hc_loss, step, epoch, N, len_loader, val_or_train, print_freq=100):
        total_loss = ce_loss + hc_loss
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))

        self.losses.update(total_loss.item(), N)
        self.hc_losses.update(hc_loss.item(), N)
        self.ce_losses.update(ce_loss.item(), N)

        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)

        if (step > 1 and step % print_freq == 0) or step == len_loader - 1:
            self.logger.info("{} : [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} CE Loss {:.3f} HC Loss {:.3f} Prec@(1, 5) ({:.1%}, {:.1%})" .format(
                    val_or_train,
                    epoch + 1,
                    self.search_epochs,
                    step,
                    len_loader - 1,
                    self.losses.get_avg(),
                    self.ce_losses.get_avg(),
                    self.hc_losses.get_avg(),
                    self.top1.get_avg(),
                    self.top5.get_avg()))


    def _reset_average_tracker(self):
        for tracker in [self.top1, self.top5, self.losses, self.ce_losses, self.hc_losses]:
            tracker.reset()

    def _evaluate_generator(self):
        evaluate_metric = {"gen_hardware_constraint":[], "target_hardware_constraint":[]}

        hc_count = 0
        total_loss = 0
        for hc in range(self.config["search_utility"]["lowest_hardware_constraint"], self.config["search_utility"]["highest_hardware_constraint"], 10):
            hc_count += 1
            target_hardware_constraint = self._get_target_hardware_constraint(hc)

            arch_param = self._get_arch_param(target_hardware_constraint)
            arch_param = self._set_arch_param(arch_param)


            arch_param_hardware_constraint = self.lookup_table.get_model_info(arch_param)
            self.logger.info(f"Generating architecture parameter hardware constraint: {arch_param_hardware_constraint}")

            hc_loss = self.hc_criterion(target_hardware_constraint, arch_param_hardware_constraint)

            evaluate_metric["gen_hardware_constraint"].append(arch_param_hardware_constraint.item())
            evaluate_metric["target_hardware_constraint"].append(hc)

            total_loss += hc_loss.item()

        kendall_tau, _ = stats.kendalltau(evaluate_metric["gen_hardware_constraint"], evaluate_metric["target_hardware_constraint"])
        total_loss /= hc_count

        return evaluate_metric, total_loss, kendall_tau


    def _save_generator_evaluate_metric(evaluate_metric):
        df_metric = pd.DataFrame(evaluate_metric)
        df_metric.to_csv(self.config["experiment_path"]["generator_evaluate_path"], index=False)
        

