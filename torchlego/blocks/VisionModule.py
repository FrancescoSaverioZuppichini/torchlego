import torch
import torch.nn as nn
from pathlib import Path
from torchsummary import summary

class VisionModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def summary(self, input_shape=(3, 224, 224)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return summary(self.to(device), input_shape)

    @classmethod
    def from_pretrain(cls, *args, **kwargs):
        # download the weights from somewhere
        pass

    @classmethod
    def from_state(cls, state_path: Path, *args, **kwargs):
        module = cls(*args, **kwargs)
        module.load_state_dict(torch.load(state_path))
        module.eval()
        return module