from torch.nn as nn
from pathlib import Path


class VisionModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder: nn.Module = None
        self.decoder: nn.Module = None

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
