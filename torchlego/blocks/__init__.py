import torch
import torch.nn as nn
from functools import partial
from .VisionModule import VisionModule
from collections import OrderedDict


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


def conv_bn(in_channels: int, out_channels: int, *args, conv=nn.Conv2d,  **kwargs):
    """
    Combines a conv layer, a batchnorm layer and an activation.
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
    blocks = OrderedDict({
        'conv': conv(in_channels, out_channels, *args, **kwargs),
        'bn':  nn.BatchNorm2d(out_channels)
    })

    return nn.Sequential(blocks)


def conv_bn_act(in_channels: int, out_channels: int, *args, conv=nn.Conv2d,  act=nn.ReLU(), **kwargs):
    """
    Combines a conv layer, a batchnorm layer and an activation.
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
    blocks = OrderedDict({
        'conv': conv(in_channels, out_channels, *args, **kwargs),
        'bn':  nn.BatchNorm2d(out_channels)
    })

    if act is not None:
        blocks['act'] = act
    return nn.Sequential(blocks)


conv1x1 = partial(nn.Conv2d, kernel_size=1)
conv3x3 = partial(nn.Conv2d, kernel_size=3)
conv1x1_bn = partial(conv_bn, conv=conv1x1)
conv3x3_bn = partial(conv_bn, conv=conv3x3)
conv1x1_bn_act = partial(conv_bn_act, conv=conv1x1)
conv3x3_bn_act = partial(conv_bn_act, conv=conv3x3)


class InputForward(nn.Module):
    """
    Pass the input to multiple modules and apply a aggregation function on the result
    """

    def __init__(self, blocks, aggr_func, *args, **kwargs):
        super().__init__()
        self.blocks = blocks
        self.aggr_func = partial(aggr_func, *args, **kwargs)

    def forward(self, x):
        res = []
        for block in self.blocks:
            out = block(x)
            res.append(out)

        return self.aggr_func(res)


Cat = partial(InputForward, aggr_func=lambda x: torch.cat(x))


class Residual(nn.Module):

    def __init__(self, blocks, res_func: callable = None, shortcut: nn.Module = nn.Identity(), *args, **kwargs):
        super().__init__()
        self.blocks = blocks
        self.shortcut = shortcut
        self.res_func = res_func

    def _module_forward(self, x):
        res = x
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = self.blocks(x)
        if self.res_func is not None:
            x = self.res_func(x, res)
        else:
            x = x, res

        return x

    def _modules_forward(self, x):
        residuals = [None] * len(self.blocks[0])

        for block in self.blocks:
            residuals = residuals[:len(block)]
            residuals.reverse()
            for i, layer in enumerate(block):
                res = residuals[i]
                if res is not None:
                    if self.shortcut is not None:
                        res = self.shortcut(res)
                    if self.res_func is not None:
                        x = self.res_func(x, res)
                    else:
                        # we haven't a .res_func, so we just pass the residual to the next layer
                        x = layer(x, res)
                else:
                    # no redisual, just pass the input
                    x = layer(x)
                residuals[i] = x

        return x

    def forward(self, x):
        if isinstance(self.blocks, nn.ModuleList):
            x = self._modules_forward(x)
        else:
            x = self._module_forward(x)
        return x


ResidualAdd = partial(Residual, res_func=lambda x, res: x + res)
ResidualCat = partial(Residual, res_func=lambda x, res: torch.cat([x, res]))
ResidualCat2d = partial(ResidualCat, res_func=lambda x,
                        res: torch.cat([x, res], dim=1))
