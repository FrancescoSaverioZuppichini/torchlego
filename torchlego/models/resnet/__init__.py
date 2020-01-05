import torch.nn as nn
from torchlego.blocks import VisionModule, ResidualAdd, conv_bn, conv_bn_act, conv3x3_bn, conv3x3_bn_act


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features, out_features, act=lambda: nn.ReLU(inplace=True), *args, **kwargs):
        super().__init__()
        shortcut = conv_bn(in_features, out_features, kernel_size=1, stride=2,
                           bias=False) if in_features != out_features else nn.Identity()
        stride = 2 if in_features != out_features else 1

        self.blocks = nn.Sequential(
            ResidualAdd(nn.ModuleList([nn.Sequential(
                conv3x3_bn_act(in_features, out_features,
                               stride=stride, padding=1, bias=False, act=act),
                conv3x3_bn(out_features, out_features,  padding=1, bias=False))]),
                shortcut=shortcut),
            act())

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_features, out_features, act=lambda: nn.ReLU(inplace=True), *args, **kwargs):
        super().__init__()
        shortcut = conv_bn(in_features, out_features * self.expansion, kernel_size=1, stride=2,
                           bias=False) if in_features != out_features else nn.Identity()
        stride = 2 if in_features != out_features else 1

        self.blocks = nn.Sequential(
            ResidualAdd(nn.ModuleList([nn.Sequential(
                conv_bn_act(in_features, out_features,
                            kernel_size=1, stride=stride, bias=False, act=act),
                conv3x3_bn_act(out_features, out_features,
                           padding=1, bias=False, act=act)),
                conv_bn(out_features, out_features * self.expansion,
                            kernel_size=1,  bias=False)

            ]),
                shortcut=shortcut),
            act())

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.blocks = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(out_features),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(
                                        kernel_size=3, stride=2, padding=1),
                                    )
    
    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNet(VisionModule):
    def __init__(self, in_features=3, n_classes=1000, block=ResNetBasicBlock, n_blocks=[3, 4, 6, 3]):
        super().__init__()
        self.encoder = nn.Sequential(
            ResNetHead(in_features, 64),
            nn.Sequential(*[block(64, 64)] * n_blocks[0]),
            block(64 * block.expansion, 128),
            nn.Sequential(*[block(128 * block.expansion, 128)]
                          * (n_blocks[1] - 1)),
            block(128 * block.expansion, 256),
            nn.Sequential(*[block(256 * block.expansion, 256)]
                          * (n_blocks[2] - 1)),
            block(256 * block.expansion, 512),
            nn.Sequential(*[block(512 * block.expansion, 512)]
                          * (n_blocks[3] - 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, n_classes)
        )

    @staticmethod
    def initialize(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        return model

    @classmethod
    def resnet18(cls, *args, **kwargs):
        return cls(*args, n_blocks=[2, 2, 2, 2], **kwargs)

    @classmethod
    def resnet34(cls, *args, **kwargs):
        return cls(*args, n_blocks=[3, 4, 6, 3], **kwargs)

    @classmethod
    def resnet50(cls, *args, **kwargs):
        return cls(*args, block=ResNetBottleNeck, n_blocks=[3, 4, 6, 3], **kwargs)

    @classmethod
    def resnet101(cls, *args, **kwargs):
        return cls(*args, block=ResNetBottleNeck, n_blocks=[3, 4, 23, 3], **kwargs)

    @classmethod
    def resnet152(cls, *args, **kwargs):
        return cls(*args, block=ResNetBottleNeck, n_blocks=[3, 8, 36, 3], **kwargs)
