
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

## Building blocks

It follows a list of useful small components made to increase your code readibily and development. Just like lego, when combined, they can become anything!

#### Convs
Most of the times you will use $3x3$ conv or $1x1$ convs followed by *batchnorm* and an *activation function*


```python
from torchlego import conv3x3

conv3x3(32, 64)
```




    Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/conv3x3.png?raw=true)


```python
from torchlego import conv3x3_bn

conv3x3_bn(32, 64)
```




    Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/conv3x3_bn.png?raw=true)


```python
from torchlego import conv3x3_bn_act

conv3x3_bn_act(32, 64)
```




    Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/conv3x3_bn_act.png?raw=true)

Optionally, you can always pass your own activation function


```python
from torch.nn import SELU
conv3x3_bn_act(32, 64, act=SELU)
```




    Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): SELU()
    )



Also, we have `conv1x1`


```python
from torchlego.blocks import conv1x1

conv1x1(32, 64)
```




    Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))



### Cat


```python
import torch
from torchlego.blocks import Cat, Lambda


blocks = [Lambda(lambda x: x), Lambda(lambda x: x)]

Cat(blocks)(torch.tensor([1]))
```




    tensor([1, 1])



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Cat.png?raw=true)

## Residual Connection

Redisual connection are a big thing. They probably are most know from *resnet* paper (even if Schmidhuber did something very similar a long time ago). You can use `torchlego` to easily create **any** residual connection that you may need.

### Residual

The main building block is the `Residual` class. Basically, it applys a function on the input and the output of a `blocks`.


```python
import torch
from torchlego.blocks import Residual, Lambda

x = torch.tensor([1])
block = Lambda(lambda x: x)


res = Residual(block, res_func=lambda x, res: x + res)
res(x)
```




    tensor([2])



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Residual.png?raw=true)

#### shortcut
We can also apply a function on the residual, this operation is called `shortcut`.


```python
res = Residual(block, res_func=lambda x, res: x + res, shortcut=lambda x: x * 2)
res(x)
```




    tensor([3])



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Residual_shorcut.png?raw=true)

#### Multiple lever residuals

If you pass an array of `nn.ModuleList`, we assume you want to pass residual trought each respective layer. In the following example all the `A` layer just compute input + 1 the input while the `B` layers add the residual with the current input. Notice that the `B.forward` function takes as second argument the residual.


```python
import torch.nn as nn

class A(nn.ModuleList):
    def forward(self, x):
        return x + 1
    
class B(nn.ModuleList):
    def forward(self, x, res):
        return x + res
        
down = nn.ModuleList([A(), A()])
up = nn.ModuleList([B(), B()])
res = Residual([down, up])
res(x)
```




    tensor([8])



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Residual_blocks.png?raw=true)

It you don't want to pass the residual in the last layer, you can just include the first `B` inside the left blocks. Be sure to handle the case where the residual is None!


```python
import torch.nn as nn


class B(nn.ModuleList):
    def forward(self, x, res = None):
        x = x if res is None else x + res
        return x 
        
down = nn.ModuleList([A(), A(), B()])
up = nn.ModuleList([B()])
res = Residual([down, up])
res(x)
```




    tensor([5])



![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Residual_blocks_no_last2.png?raw=true)


```python
import torch
from torchlego.blocks import ResidualAdd, Lambda

layer = ResidualAdd([Lambda(lambda x: x)])
layer(torch.tensor([1]))
```

A more complete example


```python
import torch.nn as nn
from torchlego.blocks import ResidualAdd

x = torch.rand((1, 64, 8, 8))

block = nn.Sequential(conv3x3_bn_act(64, 64, padding=1))

layer = ResidualAdd([block])
x = layer(x)
```

![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Add_block.png?raw=true)

You can pass multiple blocks


```python
from torchlego.blocks import ResidualAdd, Lambda

blocks = [Lambda(lambda x: x), Lambda(lambda x: x)]
layer = ResidualAdd(blocks)
layer(torch.tensor(1))
```

![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Add_blocks.png?raw=true)

Let's create a basic [ResNet](https://arxiv.org/abs/1512.03385) block



```python
from torchlego.blocks import conv_bn

def resnet_basic_block(in_features, out_features):
    shortbut = conv_bn(in_features, out_features, kernel_size=1, stride=2, bias=False) if in_features != out_features else nn.Identity()
    stride = 2 if in_features != out_features else 1
    return nn.Sequential(
                ResidualAdd(nn.ModuleList([
                    nn.Sequential(
                        conv3x3_bn_act(in_features, out_features, stride=stride, padding=1, bias=False),
                        conv3x3_bn(out_features, out_features, padding=1, bias=False))]), 
                    shortcut=shortbut),
                nn.ReLU())
    
resnet_basic_block(32, 64)
```

![alt](https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Add_resnet.png?raw=true)

What about a full `resnet`? Easy peasy


```python
def resnet(in_features, n_classes, sizes):
    return nn.Sequential(
        nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Sequential(*[resnet_basic_block(64, 64) for _ in range(sizes[0])]),
        resnet_basic_block(64, 128),
        nn.Sequential(*[resnet_basic_block(128, 128) for _ in range(sizes[1] - 1)]),
        resnet_basic_block(128, 256),
        nn.Sequential(*[resnet_basic_block(256, 256) for _ in range(sizes[2] - 1)]),
        resnet_basic_block(256, 512),
        nn.Sequential(*[resnet_basic_block(512, 512) for _ in range(sizes[3] - 1)]),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, n_classes)
)
x = torch.rand((1,3,224,244))

resnet34 = resnet(3, 1000, [3, 4, 6, 3])

resnet34(x).shape
```

![alt](https://www.researchgate.net/profile/Aaron_Vose/publication/330400293/figure/fig6/AS:715395283558403@1547574935970/ResNet-neural-network-architecture-ResNet-34-pictured-image-from-11.ppm)

### Unet


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlego.blocks import ResidualCat2d, Lambda
from torchlego.blocks import Residual, conv3x3, conv3x3_bn_act, conv1x1

down = lambda in_features, out_features: nn.Sequential(
    nn.MaxPool2d(kernel_size=2, stride=2),
    conv3x3_bn_act(in_features, out_features, padding=1),
    conv3x3_bn_act(out_features, out_features, padding=1),
)

class up(nn.Module):
    def __init__(self, in_features, out_features, should_up=True, *args, **kwargs):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_features, out_features, kernel_size=2, stride=2)
        self.should_up = should_up
        self.blocks =nn.Sequential(
            conv3x3(out_features * 2, out_features, padding=1),
            conv3x3(out_features, out_features, padding=1),
    )

    def forward(self, x, res):
        if self.should_up: x = self.up(x)
        print(res.shape)
            
        diffX = x.size()[2] - res.size()[2]
        diffY = x.size()[3] - res.size()[3]
        pad = (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2))
        res = F.pad(res, pad)
        
        x = torch.cat([res, x], dim=1)
        out = self.blocks(x)
        return out

unet = nn.Sequential(
    Residual([
    nn.ModuleList([
        nn.Sequential(
            conv3x3_bn_act(3, 64),
            conv3x3_bn_act(64, 64)),
        down(64, 128),
        down(128, 256),
        down(256, 512),
        down(512, 1024),
    ]),
    nn.ModuleList([
        up(512 * 2, 512),
        up(256 * 2, 256),
        up(128 * 2, 128),
        up(64 * 2, 64),

    ])
]),         
    conv1x1(64, 2)
)

x = torch.rand((1,3,256,256))

unet(x)
```


```python

```


```python

```
