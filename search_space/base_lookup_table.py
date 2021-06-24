import os
import sys
import time
import json

from abc import abstractmethod

import torch
import torch.nn as nn

from .block_builder import get_block
from utils import FLOPS_Counter


class LookUpTable:
    """ The abstract class of the information lookup table.
    """
    def __init__(
        self,
        macro_cfg,
        micro_cfg,
        macro_len,
        micro_len,
        table_path,
        input_size,
        info_metric="flops"):

        self.macro_cfg = macro_cfg
        self.micro_cfg = micro_cfg
        
        self.macro_len = macro_len
        self.micro_len = micro_len
        
        self.input_size = input_size

        if os.path.isfile(table_path):
            with open(table_path) as f:
                self.info_table = json.load(f)
        else:
            base_info_table = self.construct_base_info_table()
            self.info_table = self.construct_info_table()
            self.info_table.update(base_info_table)
            with open(table_path, "w") as f:
                json.dump(self.info_table, f)

        self.info_metric = info_metric

    def get_model_info(self, architecture_parameter):
        """ Calculate the model information based on the architecture parameter.

        Args:
            architecture_parameter (torch.tensor) : The vector or matrix.

        Return:
            (torch.tensor)
        """
        if len(architecture_parameter.shape) == 1:
            # Get one dim vector, convert to one-hot architecture parameter
            architecture_parameter = self._architecture_to_one_hot(
                architecture_parameter)
        else:
            architecture_parameter = architecture_parameter.reshape(self.macro_len, self.micro_len)
        model_info = []
        for i, l_ap in enumerate(architecture_parameter):
            model_info.extend(
                [p * block_info for p, block_info in zip(l_ap, self.info_table[self.info_metric][i])])


        return sum(model_info) + self.info_table["base_{}".format(self.info_metric)]
    
    def construct_base_info_table(self, info_metric_list=["flops", "param", "latency"]):
        """ Construct the table of the base information. (e.g., first stage and last stage in macro_cfg).
        
        Args:
            info_metric_list (list)

        Return:
            base_info_table (dict)
        """
        input_size = self.input_size
        base_info = 0
        base_info_table = {"base_{}".format(
            metric): 0 for metric in info_metric_list}

        first_stage = []
        first_in_channels = None
        global_stride = 1
        for l, l_cfg in enumerate(self.macro_cfg["first"]):
            block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs = l_cfg
            global_stride *= stride
            first_in_channels = in_channels if first_in_channels is None else first_in_channels
            layer = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              bn_momentum=0.1,
                              bn_track_running_stats=True,
                              **kwargs)
            first_stage.append(layer)

        first_stage = nn.Sequential(*first_stage)
        base_info = self._get_block_info(first_stage, first_in_channels, input_size, info_metric_list)
        input_size = input_size if global_stride == 1 else input_size // global_stride
        for k, v in base_info.items():
            base_info_table["base_{}".format(k)] += v

        for l, l_cfg in enumerate(self.macro_cfg["search"]):
            in_channels, out_channels, stride = l_cfg
            input_size = input_size if stride == 1 else input_size // 2
        
        last_stage = []
        last_in_channels = None
        for l, l_cfg in enumerate(self.macro_cfg["last"]):
            block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs = l_cfg
            last_in_channels = in_channels if last_in_channels is None else last_in_channels

            layer = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              bn_momentum=0.1,
                              bn_track_running_stats=True,
                              **kwargs)
            last_stage.append(layer)
        last_stage = nn.Sequential(*last_stage)
        base_info = self._get_block_info(
            last_stage, last_in_channels, input_size, info_metric_list)
        input_size = input_size if stride == 1 else input_size // 2

        for k, v in base_info.items():
            base_info_table["base_{}".format(k)] += v

        return base_info_table

    def get_search_input_size(self):
        """ Get the input size of search stage in macro_cfg

        Return:
            input_size (int): The input size of search stage
        """
        global_stride = 1
        for l, l_cfg in enumerate(self.macro_cfg["first"]):
            block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs = l_cfg
            global_stride *= stride
        input_size = self.input_size if global_stride == 1 else self.input_size // global_stride

        return input_size
        

    @abstractmethod
    def construct_info_table(self, info_metric_list=["flops", "param", "latency"]):
        """ Construct the info lookup table of search stage in macro config.

        We provide serveral useful method to calculate the info metric and process info
        metric table. Please refer to `/model/base_lookup_table.py` for more details.

        Args:
            info_metric_list (list):

        Return:
            info_table (dict)
        """
        raise NotImplemented

    def _merge_info_table(self, info_table, new_info, info_metric_list):
        """ Merge a new dict into main dict.
        
        Args:
            info_table (dict)
            new_info (dict)

        Return:
            info_table (dict)
        """
        for metric in info_metric_list:
            info_table[metric].append(new_info[metric])

        return info_table

    def _get_block_info(self, block, in_channels, input_size, info_metric_list):
        """ Calculate block information.

        Args:
            block (nn.Module)
            in_channels (int)
            input_size (int)
            info_metric_list (list)

        Return:
            block_info (dict)
        """
        block_info = {}
        for metric in info_metric_list:
            calculate_info = getattr(sys.modules[__name__], f"calculate_{metric}")
            block_info[metric] = calculate_info(block, in_channels, input_size)

        return block_info

    def _architecture_to_one_hot(self, architecture):
        """ Transfer the vector architecture index into one-hot architecture_parameter matrix.

        Args:
            architecture (np.ndarray or torch.tensor or list): The vector that store the block index of each layer.
        
        Return:
            architecture_parameter (torch.tensor): The matrix of one-hot encoding.
        """
        architecture_parameter = torch.zeros(self.macro_len, self.micro_len)
        for l, a in enumerate(architecture):
            architecture_parameter[l, a] = 1

        return architecture_parameter


def calculate_latency(model, in_channels, input_size):
    input_sample = torch.randn((1, in_channels, input_size, input_size))

    start_time = time.time()
    model(input_sample)
    latency = time.time() - start_time
    return latency


def calculate_param(model, in_channels, input_size):
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    return total_params


def calculate_flops(model, in_channels, input_size):
    if sum(p.numel() for p in model.parameters()) == 0:
        # Do not calculate flops for skip connection
        return 0

    counter = FLOPS_Counter(model, [1, in_channels, input_size, input_size])
    flops = counter.print_summary()["total_gflops"] * 1000
    return flops

def calculate_model_efficient(model, in_channels, input_size, logger):
    flops = calculate_flops(model, in_channels, input_size)
    param_nums = calculate_param(model, in_channels, input_size)
    latency = calculate_latency(model, in_channels, input_size)

    logger.info("Model efficient calculating =====================")
    logger.info(f"FLOPs : {flops}M")
    logger.info(f"Parameter number : {param_nums}")
    logger.info(f"Latency : {latency:.5f}")
    logger.info("==================================================")
