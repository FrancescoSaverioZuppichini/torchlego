import torch.nn as nn
import torch
from torchvision.models import resnet34
from torchlego.models.resnet import ResNet
from torchlego.utils import ModuleTransfer, Tracker
from torchsummary import summary


x = torch.zeros((1, 3, 224, 244))

src_model = resnet34(True).eval().cpu()
dest_model = ResNet.resnet34().eval()

x = torch.zeros((1, 3, 224, 244))

ModuleTransfer(src_model, dest_model)(x)

print(src_model(x)[0][:10])
print(dest_model(x)[0][:10])