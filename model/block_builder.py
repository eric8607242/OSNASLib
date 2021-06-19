import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_utils import IRBlock, ShuffleBlock, ShuffleBlockX, GlobalAveragePooling, ConvBNAct, MixConv, ASPPModule, SepConvBNAct, Zero

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

def _get_mixconv_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """ Construct the mixconv block of MixNet
    """
    kernel_size_list = kwargs["kernel_size_list"]
    expansion_rate = kwargs["expansion_rate"]
    point_group = kwargs["point_group"] if "point_group" in kwargs else 1

    block = MixConv(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size_list=kernel_size_list,
                    stride=stride,
                    activation=activation,
                    se=se,
                    expansion_rate=expansion_rate,
                    bn_momentum=bn_momentum,
                    bn_track_running_stats=bn_track_running_stats,
                    point_group=point_group)

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
    dilation = kwargs["dilation"] if "dilation" in kwargs else 1
    padding = kwargs["padding"] if "padding" in kwargs else (kernel_size // 2)

    block = ConvBNAct(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      activation=activation,
                      bn_momentum=bn_momentum,
                      bn_track_running_stats=bn_track_running_stats,
                      bn=bn,
                      group=group,
                      padding=padding,
                      dilation=dilation)
    return block

def _get_sepconv_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    """
    Construct the convolution block
    """
    group = kwargs["group"] if "group" in kwargs else 1
    bn = kwargs["bn"] if "bn" in kwargs else True
    dilation = kwargs["dilation"] if "dilation" in kwargs else 1
    padding = kwargs["padding"] if "padding" in kwargs else (kernel_size // 2)

    block = SepConvBNAct(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      activation=activation,
                      bn_momentum=bn_momentum,
                      bn_track_running_stats=bn_track_running_stats,
                      bn=bn,
                      group=group,
                      padding=padding,
                      dilation=dilation)
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
                          padding=0)
    else:
        if stride != 1:
            block = nn.AvgPool2d(stride, stride=stride)
        else:
            block = nn.Sequential()
    return block

def _get_asppmodule_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    padding_list = kwargs["padding_list"]
    dilation_list = kwargs["dilation_list"]

    block = ASPPModule(in_channels_list=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       padding_list=padding_list,
                       dilation_list=dilation_list,
                       bn_momentum=bn_momentum,
                       bn_track_running_stats=bn_track_running_stats)
    return block

def _get_avgpool_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    padding = kwargs["padding"] if "padding" in kwargs else (kernel_size // 2)

    block = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False)
    return block

def _get_maxpool_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    padding = kwargs["padding"] if "padding" in kwargs else (kernel_size // 2)

    block = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    return block

def _get_zero_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):
    block = Zero(in_channels=in_channels,
                 out_channels=out_channels,
                 stride=stride)
    return block


if __name__ == "__main__":
    from block_utils import IRBlock, ShuffleBlock, ShuffleBlockX, GlobalAveragePooling, ConvBNAct, MixConv, ASPPModule
    block = get_block(block_type="asppmodule",
                      in_channels=[128, 128],
                      out_channels=40,
                      kernel_size=3,
                      stride=[],
                      activation=None,
                      se=False,
                      bn_momentum=0.1,
                      bn_track_running_stats=True,
                      **{"padding_list":[1, 2],
                         "dilation_list":[2, 2]})
