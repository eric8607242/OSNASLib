import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_optimizer
from criterion import get_hc_criterion

from .base import BaseSearcher

class ArchitectureGeneratorSearcher(BaseSearcher):
    def __init__(self, supernet, val_loader, lookup_table, training_strategy, device, logger):
        super(ArchitectureGeneratorSearcher, self).__init__(supernet, val_loader, lookup_table, training_strategy, device, logger)

        generator = Generator(self.config["generator"]["hc_dim"])
        g_optimizer = get_optimizer(generator.parameters(),
                        self.config["arch_optim"]["a_optimizer"],
                        learning_rate=self.config["arch_optim"]["a_lr"],
                        weight_decay=self.config["arch_optim"]["a_weight_decay"],
                        logger=self.logger,
                        momentum=self.config["arch_optim"]["a_momentum"],
                        alpha=self.config["arch_optim"]["a_alpha"],
                        beta=self.config["arch_optim"]["a_beta"])

        self.hc_criterion = get_hc_criterion(self.config["arch_optim"]["hc_weight"])

        self.arch_param_nums = ...
        self.prior_pool = PriorPool(self.lookup_table, self.config)

    def search(self):
        search_epochs = self.config["generator"]["epochs"]
        tau = 5
        for epoch in range(search_epochs):
            self.logger.info(f"Start to train the architecture generator for epoch {epoch}")
            self.logger.info(f"Tau : {tau}")

            self._generator_training_step()
            top1_avg = self._generator_validate()

            evaluate_metric, total_loss, kendall_tau = self._evaluate_generator()

            tau *= self.config["generator"]["tau_decay"]


    def _generator_training_step(self):
        for step, (X, y) in enumerate(self.val_loader):
            self.g_optimizer.zero_grad() 
            target_hardware_constraint = self._get_target_hardware_constraint()

            arch_param = self._get_arch_param(target_hardware_constraint)
            arch_param = self._set_arch_param()

            arch_param_metric = 
            self.logger.info(f"Generating architecture parameter metric: {arch_param_metric}")

            hardware_constraint_loss = 

            X, y
            N = X.shape[0]

            ce_loss = 
            total_loss = ce_loss + hardware_constraint_loss
            self.logger.info(f"Hardware loss : {hardware_constraint_loss}")

            total_loss.backward()

            self.g_optimizer.step()


    def _generator_validate(self):
        with torch.no_grad():
            for step, (X, y) in enumerate(self.val_loader):
                pass

    def _get_target_hardware_constraint(self, target_hardware_constraint=None):
        if target_hardware_constraint is None:
            target_hardware_constraint = self.hardware_constraint_pool[self.hardware_constraint_index] + random.random() - 0.5

            self.hardware_constraint_index += 1
            if self.hardware_constraint_index == len(self.hardware_constraint_pool):
                self.hardware_constraint_index = 0
                random.shuffle(self.hardware_constraint_pool)

        target_hardware_constraint = torch.tensor(target_hardware_constraint, dtype=torch.float32).view(-1, 1)

        return target_hardware_constraint

    def set_arch_param(self, arch_param, tau=None):
        if tau is not None:
            pass
        else:
            pass

        arch_param = arch_param.to(self.device)
        return arch_param

    def _get_arch_param(self):
        hardware_constraint = target_hardware_constraint.to(self.device)
        self.logger.info(f"Target metric : {hardware_constraint.item()}")

        prior = self.prior_pool.get_prior(hardware_constraint.item())
        prior = prior.to(self.device)

        normalize_hardware_constraint = min_max_normalize()

        arch_param = self.generator(prior, normalize_hardware_constraint)
        return arch_param

    def _evaluate_generator(self):
        pass




class PriorPool:
    def __init__(self, lookup_table, arch_param_nums, config):
        self.config = config

        self.logger.info("================= Prior Pool ====================")
        self.prior_pool = self._generate_prior_pool()

    def _generate_prior_pool(self):
        prior_pool = {}

        low_info_metric = self.config
        high_info_metric = self.config

        pool_interval = (high_info_metric - low_info_metric) // (self.config)

        for metric in range(low_info_metric+pool_interval, high_info_metric, pool_interval):
            generate_metric, arch_param = self._generate_arch_param()

            while generate_metric > metric + bias or \
                    generate_metric < metric - bias:
                generate_metric, arch_param = self._generate_arch_param()

            prior_pool[str(metric)] = arch_param.tolist()
            self.logger.info(f"Target metric : {metric}, Prior generate : {generate_metric}")

    def _generate_arch_param(self):
        arch_param = torch.empty(self.marco_len, self.micro_len)

        for i in range(len(arch_param)):
            arch_param[i][random.randint(0, self.micro_len-1)] = 1

        metric = self.lookup_table.get_model_info(arch_param)
        return metric, arch_param

    def get_probability_arch_param(self, arch_param, tau):
        p_arch_param = torch.zeros_like(arch_param)

        for l_num, (l, p_l) in \
                enumerate(zip(arch_param, p_arch_param)):
            p_l = F.gumbel_softmax(l, tau=tau)
        return p_arch_param

    def get_validation_arch_param(self, arch_param):
        val_arch_param = torch.zeros_like(arch_param)
        
        for l_num, (l, v_l) in \
                enumerate(zip(arch_param, val_arch_param)):
            v_l = self._get_one_hot_vector(l)

        return val_arch_param

    def _get_one_hot_vector(self, arch_param):
        one_hot_vector = torch.zeros_like(arch_param)
        one_hot_vector[arch_param.argmax()] = 1

        return one_hot_vector
                






class Generator(nn.Module):
    def __init__(self, hc_dim, layer_nums=5, hidden_dim=32):
        super(Generator, self).__init__()
        N = hidden_dim
        self.hc_dim = hc_dim

        self.hc_head = nn.Sequential(
                ConvReLU(hc_dim, N, 3, 1, 1),
                ConvReLU(N, N, 3, 1, 1),
                ConvReLU(N, N, 3, 1, 1))
        self.prior_head = nn.Sequential(
                ConvReLU(1, N, 3, 1, 1),
                ConvReLU(N, N, 3, 1, 1))

        sel.head = ConvReLU(N, N, 3, 1, 1)

        self.main_stages = nn.Sequential()
        for i in range(layer_nums - 2):
            block = ConvReLU(N, N, 3, 1, 1)
            self.main_stages.add_module(f"block{i}", block)

        self.tail = nn.Sequential(
                nn.Conv2d(N, 1, 3, 1, 1))

        self._initialize_weights()

    def forward(self, prior):
        reshape_prior = prior.view(1, 1, *prior.shape)

        hc = hc.expand(*hc.shape[:-1], self.hc_dim * prior.size(-1), prior.size(-2))
        hc = hc.view(1, self.hc_dim, *prior.shape[-2:])

        hc = self.hc_head(hc)
        reshape_prior = self.prior_head(reshape_prior)
        hc_prior_head = hc + reshape_prior

        hc_prior = self.head(hc_prior_head)
        hc_prior = self.main_stages(hc_prior)
        hc_prior = hc_prior + hc_prior_haed

        hc_prior = self.tail(hc_prior)
        return hc_prior

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()



class ConvReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad):
        super(ConvReLU, self).__init__()

        self.add_module("conv", nn.Conv2d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          pad,
                                          bias=False))
        self.add_module("relu", nn.ReLU())
