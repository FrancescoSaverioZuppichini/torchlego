import torch
import torch.nn as nn
from functools import partial


class Conv2dPad(nn.Conv2d):

    def __init__(self, *args, mode='auto', **kwargs):
        super().__init__(*args, **kwargs)
        if mode == 'auto':
            # dynamic add padding based on the kernel_size
            self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


def conv_bn(in_channels, out_channels, *args, conv=Conv2dPad, **kwargs):
    return nn.Sequential(
        conv(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels)
    )


def conv_bn_act(*args, act=nn.ReLU(), **kwargs):
    return nn.Sequential(
        *conv_bn(*args, **kwargs),
        act
    )


conv1x1 = partial(Conv2dPad, kernel_size=1)
conv3x3 = partial(Conv2dPad, kernel_size=3)
conv3x3_bn = partial(conv_bn, conv=conv3x3)
conv3x3_bn_act = partial(conv_bn_act, conv=conv3x3)


class Residual(nn.Module):
    ADD = 'add'
    CAT = 'cat'

    def __init__(self, blocks, mode='add', shortcut=nn.Identity(), res_func=None, *args, **kwargs):
        super().__init__()
        self.blocks = blocks
        self.shortcut = shortcut
        if res_func is not None:
            self.res_func = res_func
        else:
            if mode == Residual.ADD:
                self.res_func = lambda x, res: x + res
            elif mode == Residual.CAT:
                self.res_func = lambda x, res: torch.cat(
                    [res, x], *args, **kwargs)

    def forward(self, x):
        for block in self.blocks:
            block_type = type(block)
            if block_type is nn.ModuleList:
                residuals = [None] * len(block)
                for res, layer in zip(residuals, block):
                    if res is not None:
                        res = self.shortcut(res)
                        x = self.res_func(x, res)
                        x = layer(x)
                    res.append(x)
            else:
                res = x
                x = block(x)
                res = self.shortcut(res)
                x = self.res_func(x, res)
        return x


Add = partial(Residual, mode='add')
Cat2d = partial(Residual, mode='cat', dim=1)


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
