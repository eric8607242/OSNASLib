import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAveragePooling(nn.Module):
    def forward(self, x):
        return x.mean(3).mean(2)


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class ConvBNAct(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 activation,
                 bn_momentum,
                 bn_track_running_stats,
                 padding,
                 bn=True,
                 group=1,
                 dilation=1,
                 *args,
                 **kwargs):

        super(ConvBNAct, self).__init__()

        assert activation in ["hswish", "relu", "prelu", None]
        assert stride in [1, 2, 4]

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=group,
                dilation=dilation,
                bias=False))
        
        if bn:
            self.add_module(
                "bn",
                nn.BatchNorm2d(
                    out_channels,
                    momentum=bn_momentum,
                    track_running_stats=bn_track_running_stats))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())
        elif activation == "prelu":
            self.add_module("prelu", nn.PReLU(out_channels))

class SepConvBNAct(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 activation,
                 bn_momentum,
                 bn_track_running_stats,
                 padding,
                 bn=True,
                 group=1,
                 dilation=1,
                 *args,
                 **kwargs):

        super(ConvBNAct, self).__init__()

        assert activation in ["hswish", "relu", "prelu", None]
        assert stride in [1, 2, 4]
        self.add_module(
            "conv_1",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                groups=group,
                dilation=dilation,
                bias=False))

        self.add_module(
            "conv_2",
            nn.Conv2d(
                in_channels,
                out_channels,
                1,
                stride,
                padding,
                groups=group,
                dilation=dilation,
                bias=False))
        
        if bn:
            self.add_module(
                "bn",
                nn.BatchNorm2d(
                    out_channels,
                    momentum=bn_momentum,
                    track_running_stats=bn_track_running_stats))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())
        elif activation == "prelu":
            self.add_module("prelu", nn.PReLU(out_channels))

class SEModule(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True)):
        super(SEModule, self).__init__()

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=in_channels // reduction,
                                      kernel_size=1,
                                      bias=True)
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(in_channels=in_channels // reduction,
                                     out_channels=in_channels,
                                     kernel_size=1,
                                     bias=True)
        self.excite_act = excite_act

    def forward(self, inputs):
        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        se_output = inputs * feature_excite_act

        return se_output


def channel_shuffle(x, groups=2):
    batch_size, c, w, h = x.shape
    group_c = c // groups
    x = x.view(batch_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, w, h)
    return x


class ShuffleBlockX(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 activation,
                 se,
                 bn_momentum,
                 bn_track_running_stats):
        super(ShuffleBlockX, self).__init__()
        branch_out_channels = out_channels // 2

        self.stride = stride
        self.use_branch = self.stride == 2 or in_channels != out_channels
        if self.use_branch:
            self.branch_1 = nn.Sequential(
                ConvBNAct(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=stride,
                    activation=None,
                    bn_momentum=bn_momentum,
                    bn_track_running_stats=bn_track_running_stats,
                    padding=(
                        3 // 2),
                    group=in_channels),
                ConvBNAct(
                    in_channels=in_channels,
                    out_channels=branch_out_channels,
                    kernel_size=1,
                    stride=1,
                    activation=activation,
                    bn_momentum=bn_momentum,
                    bn_track_running_stats=bn_track_running_stats,
                    padding=0))
        else:
            self.branch_1 = nn.Sequential()

        branch_2 = []
        branch_2_in_channels = in_channels if self.use_branch else branch_out_channels
        branch_2_out_channels = [[branch_2_in_channels, branch_out_channels], 
                                 [branch_out_channels, branch_out_channels], 
                                 [branch_out_channels, branch_out_channels]]
        
        for i, (oc1, oc2) in enumerate(branch_2_out_channels):
            branch_2.append(
                ConvBNAct(in_channels=oc1,
                          out_channels=oc1,
                          kernel_size=3,
                          stride=stride if i == 0 else 1,
                          activation=None,
                          bn_momentum=bn_momentum,
                          bn_track_running_stats=bn_track_running_stats,
                          padding=(3 // 2),
                          group=oc1))
            branch_2.append(
                ConvBNAct(in_channels=oc1,
                          out_channels=oc2,
                          kernel_size=1,
                          stride=1,
                          activation=activation,
                          bn_momentum=bn_momentum,
                          bn_track_running_stats=bn_track_running_stats,
                          padding=0)
            )
        self.branch_2 = nn.Sequential(*branch_2)

    def forward(self, x):
        if not self.use_branch:
            c = x.size(1) // 2
            x1, x2 = x[:, :c], x[:, c:]
            out = torch.cat((x1, self.branch_2(x2)), 1)
        else:
            out = torch.cat((self.branch_1(x), self.branch_2(x)), 1)
        return channel_shuffle(out)


class ShuffleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 activation,
                 se,
                 bn_momentum,
                 bn_track_running_stats):
        super(ShuffleBlock, self).__init__()
        branch_out_channels = out_channels // 2

        self.stride = stride
        self.use_branch = self.stride == 2 or in_channels != out_channels
        if self.use_branch:
            self.branch_1 = nn.Sequential(
                ConvBNAct(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=None,
                    bn_momentum=bn_momentum,
                    bn_track_running_stats=bn_track_running_stats,
                    padding=(
                        kernel_size // 2),
                    group=in_channels),
                ConvBNAct(
                    in_channels=in_channels,
                    out_channels=branch_out_channels,
                    kernel_size=1,
                    stride=1,
                    activation=activation,
                    bn_momentum=bn_momentum,
                    bn_track_running_stats=bn_track_running_stats,
                    padding=0))
        else:
            self.branch_1 = nn.Sequential()

        self.branch_2 = nn.Sequential(
            ConvBNAct(
                in_channels=in_channels if self.use_branch else branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=1,
                stride=1,
                activation=activation,
                bn_momentum=bn_momentum,
                bn_track_running_stats=bn_track_running_stats,
                padding=0),
            ConvBNAct(
                in_channels=branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=None,
                bn_momentum=bn_momentum,
                bn_track_running_stats=bn_track_running_stats,
                padding=(
                    kernel_size // 2),
                group=branch_out_channels),
            ConvBNAct(
                in_channels=branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=1,
                stride=1,
                activation=activation,
                bn_momentum=bn_momentum,
                bn_track_running_stats=bn_track_running_stats,
                padding=0))

    def forward(self, x):
        if not self.use_branch:
            c = x.size(1) // 2
            x1, x2 = x[:, :c], x[:, c:]
        else:
            x1, x2 = x, x
        y = torch.cat((self.branch_1(x1), self.branch_2(x2)), 1)
        return channel_shuffle(y)


class IRBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 activation,
                 se,
                 expansion_rate,
                 bn_momentum,
                 bn_track_running_stats,
                 point_group):
        super(IRBlock, self).__init__()

        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_channel = int(in_channels * expansion_rate)

        if expansion_rate == 1:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNAct(
                in_channels=in_channels,
                out_channels=hidden_channel,
                kernel_size=1,
                stride=1,
                activation=activation,
                bn_momentum=bn_momentum,
                bn_track_running_stats=bn_track_running_stats,
                group=point_group,
                padding=0)
        self.depthwise = ConvBNAct(
            in_channels=hidden_channel,
            out_channels=hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            bn_momentum=bn_momentum,
            bn_track_running_stats=bn_track_running_stats,
            padding=(
                kernel_size // 2),
            group=hidden_channel)

        self.point_wise_1 = ConvBNAct(
            in_channels=hidden_channel,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            activation=None,
            bn_momentum=bn_momentum,
            bn_track_running_stats=bn_track_running_stats,
            group=point_group,
            padding=0)

        self.se = SEModule(hidden_channel) if se else nn.Sequential()

    def forward(self, x):
        y = self.point_wise(x)
        y = self.depthwise(y)
        y = self.se(y)
        y = self.point_wise_1(y)

        y = y + x if self.use_res_connect else y

        return y

class MixConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size_list,
                 stride,
                 activation,
                 se,
                 expansion_rate,
                 bn_momentum,
                 bn_track_running_stats,
                 point_group=1):
        """ Construct mix convolution.

        As IRBlock, we first expand the feature map with the expansion rate.
        And the expansion hidden channel size will be split into block_in_channels
        based on the length of kernel size, which allow more fine-grained mixing convolution.
        """
        super(MixConv, self).__init__()

        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_channel = int(in_channels * expansion_rate)

        if expansion_rate == 1:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNAct(
                in_channels=in_channels,
                out_channels=hidden_channel,
                kernel_size=1,
                stride=1,
                activation=activation,
                bn_momentum=bn_momentum,
                bn_track_running_stats=bn_track_running_stats,
                group=point_group,
                padding=0)

        self.block_in_channels = hidden_channel // len(kernel_size_list)
        self.block_out_channels = hidden_channel // len(kernel_size_list)
        self.mixconv = nn.ModuleList()
        for kernel_size in kernel_size_list:
            operation = nn.Conv2d(self.block_in_channels,
                                  self.block_out_channels,
                                  kernel_size,
                                  stride=stride,
                                  padding=(kernel_size // 2),
                                  groups=self.block_out_channels,
                                  bias=False)
            self.mixconv.append(operation)

        self.bn_act = nn.Sequential()
        self.bn_act.add_module("bn",
                nn.BatchNorm2d(hidden_channel, 
                    momentum=bn_momentum, 
                    track_running_stats=bn_track_running_stats))

        if activation == "relu":
            self.bn_act.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.bn_act.add_module("hswish", HSwish())
        elif activation == "prelu":
            self.bn_act.add_module("prelu", nn.PReLU(out_channels))
        
        self.point_wise_1 = ConvBNAct(
            in_channels=hidden_channel,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            activation=None,
            bn_momentum=bn_momentum,
            bn_track_running_stats=bn_track_running_stats,
            group=point_group,
            padding=0)

        self.se = SEModule(hidden_channel) if se else nn.Sequential()

    def forward(self, x):
        y = self.point_wise(x)

        # Mix conv block ====================
        split_y = torch.split(y, self.block_in_channels, dim=1)
        y = torch.cat([block_i(y_i) for y_i, block_i in zip(split_y, self.mixconv)], dim=1)
        y = self.bn_act(y)
        # ==================================

        y = self.se(y)
        y = self.point_wise_1(y)

        y = y + x if self.use_res_connect else y
        return y

class ASPPModule(nn.Module):
    def __init__(self,
                 in_channels_list,
                 out_channels,
                 kernel_size,
                 padding_list,
                 dilation_list,
                 bn_momentum,
                 bn_track_running_stats):
        super(ASPPModule, self).__init__()

        self.aspps = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list):
            self.aspps.append(ASPP(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding_list[i],
                                   dilation=dilation_list[i],
                                   bn_momentum=bn_momentum,
                                   bn_track_running_stats=bn_track_running_stats))

    def forward(self, xs):
        first_y = self.aspps(xs[0])
        first_y = F.upsample(first_y, size=(320, 320), mode="bilinear", align_corners=True)

        y_list = [first_y]
        for i in range(1, len(self.aspps)):
            y = self.aspps[i](xs[i])
            y = F.upsample(y, size=(320, 320), mode="bilinear", align_corners=True)
            
            y_list.append(y)

        return sum(y_list)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, bn_momentum, bn_track_running_stats):
        super(ASPP, self).__init__()
        self.conv_1 = ConvBNAct(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                activation=None,
                                bn_momentum=bn_momentum,
                                bn_track_running_stats=bn_track_running_stats,
                                padding=0)
        self.conv_2 = ConvBNAct(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                activation=None,
                                bn_momentum=bn_momentum,
                                bn_track_running_stats=bn_track_running_stats,
                                padding=padding,
                                dilation=dilation)
        self.conv_3 = ConvBNAct(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                activation=None,
                                bn_momentum=bn_momentum,
                                bn_track_running_stats=bn_track_running_stats,
                                padding=0)

        self.concate_conv = nn.Conv2d(in_channels*3,
                                      out_channels,
                                      1,
                                      1,
                                      0)

    def forward(self, x):
        y_1 = self.conv_1(x)
        y_2 = self.conv_2(y_1)

        y_3 = F.avg_pool2d(x, kernel_size=x.size()[2:])
        y_3 = F.upsample(y_3, size=x.size()[2:], align_corners=True, mode="bilinear")
        y_3 = self.conv_3(y_3)

        y = torch.cat([y_1, y_2, y_3], dim=1)
        y = self.concate_conv(y)

        return y



class Zero(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Zero, self).__init__()
        self.stride = stride

        self.conv_flag = False
        if in_channels != out_channels:
            self.conv_flag = True
            self.zero_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        if self.conv_flag:
            return self.zero_conv(x)
        return x[:,:,::self.stride,::self.stride].mul(0.)
           


                

