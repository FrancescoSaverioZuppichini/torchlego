from dataclasses import dataclass, field
import torch.nn as nn
import copy

@dataclass
class Tracer:
    module: nn.Module
    traced: [nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs, outputs):
        if len(list(m.modules())) == 1:
            self.traced.append(m)

    def __call__(self, x):
        for m in self.module.modules():
            self.handles.append(
                m.register_forward_hook(self._forward_hook))
        self.module(x)
        list(map(lambda x: x.remove(), self.handles))
        return self

    @property
    def parametrized(self):
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module

    def __call__(self, x):
        dest_traced = Tracer(self.dest)(x).parametrized
        src_traced = Tracer(self.src)(x).parametrized
        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            print(f'Transfered from={src_m} to={dest_m}')
            for key in src_m.state_dict().keys():
                assert dest_m.state_dict()[key].sum() == src_m.state_dict()[key].sum()
                
        print('=======================================================')
        
        for dest_m, src_m in zip(dest_traced, src_traced):
            print(dest_m)
            for key in dest_m.state_dict().keys():
                assert dest_m.state_dict()[key].sum() == src_m.state_dict()[key].sum()
