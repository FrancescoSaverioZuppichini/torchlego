import torch
import torch.nn as nn
from functools import partial
from .VisionModule import VisionModule

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Conv2dPad(nn.Conv2d):
    """
    Replacement for nn.Conv2d with padding by default.

    :Example:

    from torchlego.blocks import Conv2dPad

    Conv2dPad(32, 64, kernel_size=3) # will apply a padding of 1
    Conv2dPad(32, 64, kernel_size=3, mode=None) # won't apply padding.
    """
    def __init__(self, *args, mode='auto', **kwargs):
        super().__init__(*args, **kwargs)
        if mode == 'auto':
            # dynamic add padding based on the kernel_size
            self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


def conv_bn(in_channels: int, out_channels:int, *args, conv=nn.Conv2d, **kwargs):
    """
    Combines a conv layer with a batchnorm layer. 
    Useful to code faster and increases readibility. 

    

    :param in_channels: [description]
    :type in_channels: int
    :param out_channels: [description]
    :type out_channels: int
    :param conv: [description], defaults to Conv2dPad
    :type conv: [type], optional
    :return: [description]
    :rtype: [type]
    """
    return nn.Sequential(
        conv(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels)
    )


def conv_bn_act(*args, act=nn.ReLU, **kwargs):
    """
    Combines a conv layer, a batchnorm layer and an activation. 
    Useful to code faster and increases readibility. 

    :param act: [description], defaults to nn.ReLU()
    :type act: [type], optional
    :return: [description]
    :rtype: [type]
    """
    # TODO if act is None -> ReLU
    return nn.Sequential(
        *conv_bn(*args, **kwargs),
        act()
    )


conv1x1 = partial(nn.Conv2d, kernel_size=1)
conv3x3 = partial(nn.Conv2d, kernel_size=3)
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
                        if self.shortcut is not None:
                            res = self.shortcut(res)
                        x = self.res_func(x, res)
                        x = layer(x)
                    res.append(x)
            else:
                res = x
                x = block(x)
                if self.shortcut is not None:
                    res = self.shortcut(res)
                x = self.res_func(x, res)
        return x


ResidualAdd = partial(Residual, mode='add')
ResidualCat = partial(Residual, mode='cat')
ResidualCat2d = partial(Residual, mode='cat', dim=1)
