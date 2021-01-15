import os 
import json

import torch
import torch.nn as nn

from utils.network_utils import get_block
from utils.countflops import FLOPS_Counter


class LookUpTable:
    def __init__(self, macro_cfg, micro_cfg, table_path, input_size, info_metric=["flops", "param", "latency"]):
        if os.path.isfile(table_path):
            with open(table_path) as f:
                self.info_table = json.load(f)
        else:
            self.info_table = self._construct_info_table(macro_cfg, micro_cfg, input_size, info_metric=info_metric)
            with open(table_path, "w") as f:
                json.dump(self.info_table, f)
        print(self.info_table)


    def get_model_info(self, architecture_parameter, info_metric="flops"):
        """
        architecture_parameter(matrix) : one-hot of the architecture
        """
        model_info = []
        for i, l_ap in enumerate(architecture_parameter):
            model_info.append(p * block_info for p, block_info in zip(l_ap, self.info_table[info_metric][i]))
        
        return sum(model_info)
    

    def _construct_info_table(self, macro_cfg, micro_cfg, input_size, info_metric=["flops", "param", "latency"]):
        info_table = {metric:[] for metric in info_metric}

        for l, l_cfg in enumerate(macro_cfg["search"]):
            in_channels, out_channels, stride = l_cfg
            layer_info = {metric:[] for metric in info_metric}
            
            for b, b_cfg in enumerate(micro_cfg):
                block_type, kernel_size, se, activation, kwargs = b_cfg
                block = get_block(block_type=block_type,
                                  in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  activation=activation,
                                  se=se,
                                  bn_momentum=0.9,
                                  bn_track_running_stats=False,
                                  **kwargs
                                )

                block_info = self._get_block_info(block, in_channels, input_size, info_metric)
                layer_info = self._merge_info_table(layer_info, block_info, info_metric)

            input_size = input_size if stride == 1 else input_size//2
            info_table = self._merge_info_table(info_table, layer_info, info_metric)

        return info_table


    def _merge_info_table(self, info_table, new_info, info_metric):
        """Merge a dict with new info in to main info_table
        info_table(dict) : Main info table
        new_info(dict)
        """
        for metric in info_metric:
            info_table[metric].append(new_info[metric])

        return info_table

    def _get_block_info(self, block, in_channels, input_size, info_metric):
        block_info = {}
        for metric in info_metric:
            if metric == "flops":
                block_info["flops"] = calculate_flops(block, in_channels, input_size)
            elif metric == "param":
                block_info["param"] = calculate_param_nums(block)
            elif metric == "latency":
                pass
            else:
                raise NotImplementedError

        return block_info


def calculate_param_nums(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def calculate_flops(model, in_channels, input_size):
    counter = FLOPS_Counter(model, [1, in_channels, input_size, input_size])
    flops = counter.print_summary()["total_gflops"]*1000
    return flops
