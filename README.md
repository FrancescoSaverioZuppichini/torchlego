# torchlego 

High quality Neural Networks built with reausable blocks in PyTorch

![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/torchlego/develop/doc/images/lego.jpg)

*Photo by Ryan Quintal on Unsplash*

## Installation

## Quick Tour


```python
%load_ext autoreload
%autoreload 2
```

### Useful building blocks

It follows a list of useful small components that can increase your code readibily and development


```python
from torchlego import conv3x3

conv3x3(32, 64)
```




    Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/conv3x3.png?raw=true)


```python
from torchlego import conv3x3_bn

conv3x3_bn(32, 64)
```




    Sequential(
      (0): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/conv3x3_bn.png?raw=true)


```python
from torchlego import conv3x3_bn_act

conv3x3_bn_act(32, 64)
```




    Sequential(
      (0): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/conv3x3_bn_act.png?raw=true)

Optionally, you can always pass your own activation function


```python
from torch.nn import SELU
conv3x3_bn_act(32, 64, act=SELU())
```




    Sequential(
      (0): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): SELU()
    )



Also, we have `conv1x1`

## Residual

Redisual connection are a big thing. You can use `torchlego` to easily create **any** residual connection that you may need.


```python
from torchlego import Add, Lambda

layer = Add([Lambda(lambda x: x)])
layer(torch.tensor(1))
```




    tensor(2)



A more complete example


```python
from torchlego import Add

x = torch.rand((1, 64, 8, 8))

block = nn.Sequential(conv3x3_bn_act(64, 64))

layer = Add([block])
x = layer(x)
```

![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Add_block.png?raw=true)

You can pass multiple blocks


```python
from torchlego import Add, Lambda

blocks = [Lambda(lambda x: x), Lambda(lambda x: x)]
layer = Add(blocks)
layer(torch.tensor(1))
```




    tensor(4)



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Add_blocks.png?raw=true)

Let's create a basic [ResNet](https://arxiv.org/abs/1512.03385) block



```python
def resnet_basic_block(in_features, out_features):
    shortbut = conv3x3_bn(in_features, out_features, stride=2, bias=False) if in_features != out_features else nn.Identity()
    stride = 2 if in_features != out_features else 1
    return nn.Sequential(
                Add(nn.ModuleList([
                    nn.Sequential(
                        conv3x3_bn_act(in_features, in_features, stride=stride, bias=False),
                        conv3x3_bn(in_features, out_features, bias=False))]), 
                    shortcut=shortbut),
                nn.ReLU())
    
resnet_basic_block(32, 64)
```




    Sequential(
      (0): Residual(
        (blocks): ModuleList(
          (0): Sequential(
            (0): Sequential(
              (0): Conv2dPad(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (1): Sequential(
              (0): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (shortcut): Sequential(
          (0): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): ReLU()
    )



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Add_resnet.png?raw=true)

What about `resnet34`? Easy peasy


```python
ResNet34 = nn.Sequential(
    Conv2dPad(3, 64, kernel_size=7, stride=4),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=1),
    nn.Sequential(*[resnet_basic_block(64, 64)] * 3),
    resnet_basic_block(64, 128),
    nn.Sequential(*[resnet_basic_block(128, 128)] * 3),
    resnet_basic_block(128, 256),
    nn.Sequential(*[resnet_basic_block(256, 256)] * 5),
    resnet_basic_block(256, 512),
    nn.Sequential(*[resnet_basic_block(512, 512)] * 2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(512, 1000)
)

x = torch.rand((1,3,224,244))

ResNet34(x).shape
```




    torch.Size([1, 1000])



![alt](https://www.researchgate.net/profile/Aaron_Vose/publication/330400293/figure/fig6/AS:715395283558403@1547574935970/ResNet-neural-network-architecture-ResNet-34-pictured-image-from-11.ppm)

### Unet


```python
# down = lambda in_features, out_features: nn.Sequential(
#     conv3x3(in_features, in_features),
#     conv3x3(in_features, out_features),
#     nn.MaxPool2d(out_features)
# )

# up = lambda in_features, out_features : nn.Sequential(
#     conv3x3(in_features, in_features),
#     conv3x3(in_features, out_features),
#     nn.ConvTranspose2d(out_features, out_features, kernel_size=2)
# )

# Concat(
#     [
#         down(64, 128),
#         down(128, 256),
#         down(256, 512),
#         down(512, 1024),
#     ],
#     [
#         up(1024, 512 * 2),
#         up(256 * 2, 128),
#         up(128 * 2, 64),
#         nn.Sequential(
#             conv3x3(64), 
#             conv3x3(64))
#     ]
# )
```
