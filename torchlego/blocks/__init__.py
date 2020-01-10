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

class Cat(nn.Module):
    """
    Simply concat all the outputs from the `.blocks` modules.
    """
    def __init__(self, blocks, *args, **kwargs):
        super().__init__()
        self.blocks = blocks
        self.cat = partial(torch.cat, *args, **kwargs)
        
    def forward(self, x):
        res =  []
        for block in self.blocks:
            out = block(x)
            res.append(out)
            
        return self.cat(res)

class Residual(nn.Module):

    def __init__(self, blocks, res_func:callable=None, shortcut:nn.Module=nn.Identity(), *args, **kwargs):
        super().__init__()
        self.blocks = blocks if type(blocks) is list or type(blocks) is nn.ModuleList else [blocks]
        self.shortcut = shortcut
        self.res_func = res_func
   
    def forward(self, x):
        residuals = [None] * (len(self.blocks[0]))
        for block in self.blocks:
            block_type = type(block)
            if block_type is nn.ModuleList:
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
                            x = layer(x, res)
                    else:
                        x = layer(x)
                    residuals[i] = x

            else:
                res = x
                x = block(x)
                if self.shortcut is not None:
                    res = self.shortcut(res)
                x = self.res_func(x, res)
        print([res.shape[1] for res in residuals])

        return x


ResidualAdd = partial(Residual, res_func=lambda x, res: x + res)
ResidualCat = partial(Residual, res_func=lambda x, res: torch.cat([x, res]))
ResidualCat2d = partial(ResidualCat, res_func=lambda x, res: torch.cat([x, res], dim=1))
