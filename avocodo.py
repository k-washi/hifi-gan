import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from dataclasses import dataclass

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

@dataclass
class AvocodoConfig:
    resblock: int = 1
    upsample_rates = [[8], [8], [2], [2]]
    upsample_kernel_sizes = [[16], [16], [4], [4]]
    upsample_initial_channel: int = 512
    resblock_kernel_sizes = [3,7,11]
    resblock_dilation_sizes =  [[1,3,5], [1,3,5], [1,3,5]]
    projection_filters =  [0, 1, 1, 1]
    projection_kernels = [0, 5, 7, 11]
    pqmf_lv1 = [2, 256, 0.25, 10.0]
    pqmf_lv2 = [4, 192, 0.13, 10.0]
    hop_size = 256
    n_fft = 512
    win_size = 512
    sr = 16000



class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.2)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.2)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for _l in self.convs1:
            remove_weight_norm(_l)
        for _l in self.convs2:
            remove_weight_norm(_l)



class Avocodo(torch.nn.Module):
    def __init__(self, cfg: AvocodoConfig):
        super(Avocodo, self).__init__()
        
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, cfg.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            _ups = nn.ModuleList()
            for _i, (_u, _k) in enumerate(zip(u, k)):
                in_channel = cfg.upsample_initial_channel // (2**i)
                out_channel = cfg.upsample_initial_channel // (2**(i + 1))
                _ups.append(weight_norm(
                    ConvTranspose1d(in_channel, out_channel, _k, _u, padding=(_k - _u) // 2)))
            self.ups.append(_ups)

        self.resblocks = nn.ModuleList()
        self.conv_post = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = cfg.upsample_initial_channel // (2**(i + 1))
            temp = nn.ModuleList()
            for j, (k, d) in enumerate(zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes)):
                temp.append(ResBlock(ch, k, d))
            self.resblocks.append(temp)

            if cfg.projection_filters[i] != 0:
                self.conv_post.append(
                    weight_norm(
                        Conv1d(
                            ch, cfg.projection_filters[i],
                            cfg.projection_kernels[i], 1, padding=cfg.projection_kernels[i] // 2
                        )))
            else:
                self.conv_post.append(torch.nn.Identity())

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        outs = []
        x = self.conv_pre(x)
        for i, (ups, resblocks, conv_post) in enumerate(zip(self.ups, self.resblocks, self.conv_post)):
            x = F.leaky_relu(x, 0.2)
            for _ups in ups:
                x = _ups(x)
            xs = None
            for j, resblock in enumerate(resblocks):
                if xs is None:
                    xs = resblock(x)
                else:
                    xs += resblock(x)
            x = xs / self.num_kernels
            if i >= (self.num_upsamples-3):
                _x = F.leaky_relu(x)
                _x = conv_post(_x)
                _x = torch.tanh(_x)
                outs.append(_x)
            else:
                x = conv_post(x)

        return outs

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for ups in self.ups:
            for _l in ups:
                remove_weight_norm(_l)
        for resblock in self.resblocks:
            for _l in resblock:
                _l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        for _l in self.conv_post:
            if not isinstance(_l, torch.nn.Identity):
                remove_weight_norm(_l)