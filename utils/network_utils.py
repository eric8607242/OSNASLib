import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x*1. / divisible_by)* divisible_by)


def conv_1x1_bn(in_channels, 
                out_channels, 
                activation,
                bn_momentum,
                bn_track_running_stats):
    if activation == "relu":
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum, track_running_stats=bn_track_running_stats),
            nn.ReLU6(inplace=True)
        )
    elif activation == "hswish":
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_channels, momentum=bn_momentum, track_running_stats=bn_track_running_stats),
            HSwish()
        )

class GlobalAveragePooling(nn.Module):
    def forward(self, x):
        return x.mean(3).mean(2)

class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True) / 6
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
                 pad,
                 group=1,
                 *args,
                 **kwargs):

        super(ConvBNAct, self).__init__()

        assert activation in ["hswish", "relu", None]
        assert stride in [1, 2, 4]

        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, groups=group, bias=False))
        self.add_module("bn", nn.BatchNorm2d(out_channels, momentum=bn_momentum, track_running_stats=bn_track_running_stats))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())

            
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
        if self.stride == 2:
            self.branch_1 = nn.Sequential(ConvBNAct(in_channels=in_channels,
                                                   out_channels=in_channels,
                                                   kernel_size=3,
                                                   stride=stride,
                                                   activation=None,
                                                   bn_momentum=bn_momentum,
                                                   bn_track_running_stats=bn_track_running_stats,
                                                   pad=(3//2),
                                                   group=in_channels),
                                          ConvBNAct(in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    activation=activation,
                                                    bn_momentum=bn_momentum,
                                                    bn_track_running_stats=bn_track_running_stats,
                                                    pad=0))
        else:
            self.branch_1 = nn.Sequential()

        branch_2 = []
        branch_2_out_channels = [[in_channels, branch_out_channels], [branch_out_channels, branch_out_channels], [branch_out_channels, branch_out_channels]]
        branch_2_in_channels = in_channels if stride == 2 else branch_out_channels
        for i, (oc1, oc2) in enumerate(branch_2_out_channels):
            branch_2.append(
                            ConvBNAct(in_channels=branch_2_in_channels,
                                      out_channels=oc1,
                                      kernel_size=3,
                                      stride=stride if i == 0 else 1,
                                      activation=None,
                                      bn_momentum=bn_momentum,
                                      bn_track_running_stats=bn_track_running_stats,
                                      pad=(3//2),
                                      group=branch_2_in_channels))
            branch_2.append(
                            ConvBNAct(in_channels=oc1,
                                      out_channels=oc2,
                                      kernel_size=1,
                                      stride=1,
                                      activation=activation,
                                      bn_momentum=bn_momentum,
                                      bn_track_running_stats=bn_track_running_stats,
                                      pad=0)
                            )
            branch_2_in_channels = oc2
        self.branch_2 = nn.Sequential(*branch_2)
            
        

    def forward(self, x):
        if self.stride == 1:
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
        if self.stride == 2:
            self.branch_1 = nn.Sequential(ConvBNAct(in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    activation=None,
                                                    bn_momentum=bn_momentum,
                                                    bn_track_running_stats=bn_track_running_stats,
                                                    pad=(kernel_size//2),
                                                    group=in_channels),
                                          ConvBNAct(in_channels=in_channels,
                                                    out_channels=branch_out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    activation=activation,
                                                    bn_momentum=bn_momentum,
                                                    bn_track_running_stats=bn_track_running_stats,
                                                    pad=0))
        else:
            self.branch_1 = nn.Sequential()

        self.branch_2 = nn.Sequential(ConvBNAct(in_channels=in_channels if stride > 1 else branch_out_channels,
                                                out_channels=branch_out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                activation=activation,
                                                bn_momentum=bn_momentum,
                                                bn_track_running_stats=bn_track_running_stats,
                                                pad=0),
                                      ConvBNAct(in_channels=branch_out_channels,
                                                out_channels=branch_out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                activation=None,
                                                bn_momentum=bn_momentum,
                                                bn_track_running_stats=bn_track_running_stats,
                                                pad=(kernel_size//2),
                                                group=branch_out_channels),
                                      ConvBNAct(in_channels=branch_out_channels,
                                                out_channels=branch_out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                activation=activation,
                                                bn_momentum=bn_momentum,
                                                bn_track_running_stats=bn_track_running_stats,
                                                pad=0))

    def forward(self, x):
        if self.stride == 1:
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
            self.point_wise = ConvBNAct(in_channels=in_channels,
                                        out_channels=hidden_channel,
                                        kernel_size=1,
                                        stride=1,
                                        activation=activation,
                                        bn_momentum=bn_momentum,
                                        bn_track_running_stats=bn_track_running_stats,
                                        group=point_group,
                                        pad=0) 
        self.depthwise = ConvBNAct(in_channels=hidden_channel,
                                   out_channels=hidden_channel,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   activation=activation,
                                   bn_momentum=bn_momentum,
                                   bn_track_running_stats=bn_track_running_stats,
                                   pad=(kernel_size//2),
                                   group=hidden_channel)

        self.point_wise_1 = ConvBNAct(in_channels=hidden_channel,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      activation=None,
                                      bn_momentum=bn_momentum,
                                      bn_track_running_stats=bn_track_running_stats,
                                      group=point_group,
                                      pad=0)

        self.se = SEModule(hidden_channel) if se else nn.Sequential()

    def forward(self, x):
        y = self.point_wise(x)
        y = self.depthwise(y)
        y = self.se(y)
        y = self.point_wise_1(y)

        y = y+x if self.use_res_connect else y

        return y

def get_block(block_type,
              in_channels,
              out_channels,
              kernel_size,
              stride,
              activation,
              se,
              bn_momentum,
              bn_track_running_stats,
              *args,
              **kwargs):

    if block_type == "Mobile":
        # Inverted Residual Block of MobileNet
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

    elif block_type == "Shuffle":
        # Block of ShuffleNet
        block = ShuffleBlock(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             activation=activation,
                             se=se,
                             bn_momentum=bn_momentum,
                             bn_track_running_stats=bn_track_running_stats)
    elif block_type == "ShuffleX":
        # Block of ShuffleNet
        block = ShuffleBlockX(in_channels=in_channels,
                              out_channels=out_channels,
                              stride=stride,
                              activation=activation,
                              se=se,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats)


    elif block_type == "classifier":
        block = nn.Linear(in_channels, out_channels)

    elif block_type == "global_average":
        block = GlobalAveragePooling()
    
    elif block_type == "Conv":
        block = ConvBNAct(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          activation=activation,
                          bn_momentum=bn_momentum,
                          bn_track_running_stats=bn_track_running_stats,
                          group=1,
                          pad=(kernel_size//2))

    elif block_type == "Skip":
        if in_channels != out_channels:
            block = ConvBNAct(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=stride,
                              activation="relu",
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              group=1,
                              pad=(3//2))
        else:
            block = nn.Sequential()

    else:
        raise NotImplementedError

    return block
    
