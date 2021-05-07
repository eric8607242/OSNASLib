import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorPool:
    def __init__(self, lookup_table, arch_param_nums, config):
        self.config = config

        self.logger.info("================= Prior Pool ====================")
        self.prior_pool = self._generate_prior_pool()

        self.marco_len = arch_param_nums[1]
        self.micro_len = arch_param_nums[0]

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
                
