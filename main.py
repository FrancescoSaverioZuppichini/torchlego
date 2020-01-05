import torch.nn as nn
import torch
from torchvision.models import resnet34
from torchlego.blocks import *
from torchsummary import summary
from torchlego.utils import ModuleTransfer, Tracer


def resnet_basic_block(in_features, out_features):
    shortcut = conv_bn(in_features, out_features, kernel_size=1, stride=2,
                       bias=False) if in_features != out_features else nn.Identity()
    stride = 2 if in_features != out_features else 1
    return nn.Sequential(
        Add(nn.ModuleList([nn.Sequential(
                conv3x3_bn_act(in_features, out_features,
                               stride=stride, padding=1, bias=False),
                conv3x3_bn(out_features, out_features,  padding=1, bias=False))]),
            shortcut=shortcut),
        nn.ReLU(inplace=True)
    )


def ResNet34():

    model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    nn.Sequential(*[resnet_basic_block(64, 64)] * 3),
    resnet_basic_block(64, 128),
    nn.Sequential(*[resnet_basic_block(128, 128)] * 3),
    resnet_basic_block(128, 256),
    nn.Sequential(*[resnet_basic_block(256, 256)] * 5),
    resnet_basic_block(256, 512),
    nn.Sequential(*[resnet_basic_block(512, 512)] * 2),
    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    nn.Flatten(),
    nn.Linear(512, 1000)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model


x = torch.zeros((1, 3, 224, 244))

resnet34_pre = resnet34(True).eval()
resnet34_my = ResNet34().eval()
print(resnet34_my)
# print(resnet34_my(x)[0][0:10])
module_stranfer = ModuleTransfer(resnet34_pre, resnet34_my)
module_stranfer(x)

# src_traced = Tracer(resnet34_pre)(x).parametrized
# dest_traced = Tracer(resnet34_my)(x).parametrized
# with torch.no_grad():
#     for dest_m, src_m in zip(dest_traced, src_traced):
#         try:
#             for key in dest_m.state_dict().keys():
#                 assert dest_m.state_dict()[key].sum() == src_m.state_dict()[key].sum()
#         except AssertionError as e:
#             print(f'Layer={dest_m} with key={key} has not the same params.')
# with torch.no_grad():
#     print(resnet34_pre(x)[0][0:10])
#     print(resnet_trans(x)[0][0:10])
# print(resnet34_my)
