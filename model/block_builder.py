import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_utils import IRBlock, ShuffleBlock, ShuffleBlockX, GlobalAveragePooling, ConvBNAct

def get_block(block_type, in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """ The factory to construct the candidate block

    Args:
        block_type (str)
        in_channels (int)
        out_channels (int)
        kernel_size (int)
        stride (int)
        activation (str)
        se (bool)
        bn_momentum (float)
        bn_track_running_stats (bool)

    Return:
        block (nn.Module)
    """
    block_method = getattr(sys.modules[__name__], f"_get_{block_type}_block")
    block = block_method(in_channels, out_channels, kernel_size, 
                stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs)

    return block


def _get_mobile_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the inverted residual block of MobileNetV2
    """
    expansion_rate = kwargs["expansion_rate"]
    point_group = kwargs["point_group"] if "point_group" in kwargs else 1

    block = IRBlock(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=activation,
                    se=se,
                    expansion_rate=expansion_rate,
                    bn_momentum=bn_momentum,
                    bn_track_running_stats=bn_track_running_stats,
                    point_group=point_group)
    return block


def _get_shuffle_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the shuffle block of ShuffleNet
    """
    block = ShuffleBlock(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         activation=activation,
                         se=se,
                         bn_momentum=bn_momentum,
                         bn_track_running_stats=bn_track_running_stats)
    return block


def _get_shuffleX_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the shuffleX block of ShuffleNet
    """
    block = ShuffleBlockX(in_channels=in_channels,
                          out_channels=out_channels,
                          stride=stride,
                          activation=activation,
                          se=se,
                          bn_momentum=bn_momentum,
                          bn_track_running_stats=bn_track_running_stats)
    return block


def _get_unified_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the unified block of SGNAS
    """
    max_expansion_rate = kwargs["max_expansion_rate"]
    min_expansion_rate = kwargs["min_expansion_rate"]

    return block


def _get_classifier_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the classifier block
    """
    block = nn.Linear(in_channels, out_channels)
    return block


def _get_global_average_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the global average block
    """
    block = GlobalAveragePooling()
    return block


def _get_conv_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the convolution block
    """
    group = kwargs["group"] if "group" in kwargs else 1
    bn = kwargs["bn"] if "bn" in kwargs else True
    block = ConvBNAct(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      activation=activation,
                      bn_momentum=bn_momentum,
                      bn_track_running_stats=bn_track_running_stats,
                      bn=bn,
                      group=group,
                      pad=(kernel_size // 2))
    return block


def _get_skip_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the skip connection block
    """
    if in_channels != out_channels:
        block = ConvBNAct(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          activation=None,
                          bn_momentum=bn_momentum,
                          bn_track_running_stats=bn_track_running_stats,
                          group=1,
                          pad=0)
    else:
        if stride != 1:
            block = nn.AvgPool2d(stride, stride=stride)
        else:
            block = nn.Sequential()
    return block
