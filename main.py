import torch.nn as nn
import torch
from torchvision.models import resnet34
from torchlego.models.resnet import ResNet
from torchlego.utils import ModuleTransfer
from torchsummary import summary


x = torch.zeros((1, 3, 224, 244))

resnet34_pre = resnet34(True).eval().cpu()
resnet34_my = ResNet.resnet34().eval().cpu()
# print(resnet34_my)
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
