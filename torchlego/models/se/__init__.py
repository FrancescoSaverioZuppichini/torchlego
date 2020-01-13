import torch.nn as nn

class SELayer(nn.Module):
    """
    [Squeeze and Excitation](https://arxiv.org/abs/1709.01507) module copied from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.blocks = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.blocks(y).view(b, c, 1, 1)
        return x * y.expand_as(x)