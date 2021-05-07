import math 

import torch
import torch.nn as nn

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

        self.head = ConvReLU(N, N, 3, 1, 1)

        self.main_stages = nn.Sequential()
        for i in range(layer_nums - 2):
            block = ConvReLU(N, N, 3, 1, 1)
            self.main_stages.add_module(f"block{i}", block)

        self.tail = nn.Sequential(
                nn.Conv2d(N, 1, 3, 1, 1))

        self._initialize_weights()

    def forward(self, prior, hc):
        reshape_prior = prior.view(1, 1, *prior.shape)

        hc = hc.expand(*hc.shape[:-1], self.hc_dim * prior.size(-1), prior.size(-2))
        hc = hc.view(1, self.hc_dim, *prior.shape[-2:])

        hc = self.hc_head(hc)
        reshape_prior = self.prior_head(reshape_prior)
        hc_prior_head = hc + reshape_prior

        hc_prior = self.head(hc_prior_head)
        hc_prior = self.main_stages(hc_prior)
        hc_prior = hc_prior + hc_prior_head

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
