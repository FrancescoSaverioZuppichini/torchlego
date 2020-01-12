from torchlego.models.resnet import ResNet
from torchvision.models import resnet50, resnet34
from torchsummary import summary
# print(resnet50(False))

print(summary(resnet34(False).cuda(), (3, 224, 224)))
print(ResNet.resnet34().summary()) #

# print(ResNet.initialize(ResNet()))