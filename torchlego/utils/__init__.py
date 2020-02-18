import torch
import copy
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class Tracker:
    """This class tracks all the operations of a given module by performing a forward pass. 

    Example: ::

    ```python3
    import torch
    import torch.nn as nn
    from torchlego.utils import
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())
    tr = Tracker(model)
    tr(x)
    print(tr.traced) # all operations
    print('-----')
    print(tr.parametrized) # all operations with learnable params
    ```

    Outputs 

    ```
    [Linear(in_features=1, out_features=64, bias=True),
    ReLU(),
    Linear(in_features=64, out_features=10, bias=True),
    ReLU()]
    -----
    [Linear(in_features=1, out_features=64, bias=True),
    Linear(in_features=64, out_features=10, bias=True)]
    ```
    """
    module: nn.Module
    traced: [nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs, outputs):
        has_not_submodules = len(list(m.modules())) == 1
        if has_not_submodules:
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
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


@dataclass
class ModuleTransfer:
    """This class transfers the weight from one module to another assuming 
    they have the same set of operations but they were defined in a different way.

    Example:: 

    ```python3

    import torch
    import torch.nn as nn
    from torchlego.utils import ModuleTransfer

    model_a = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())

    def block(in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features),
                            nn.ReLU())

    model_b = nn.Sequential(block(1,64), block(64,10))

    # model_a and model_b are the same thing but defined in two different ways
    x = torch.ones(1, 1)

    trans = ModuleTransfer(src=model_a, dest=model_b)
    trans(x)
    # now module_b has the same weight of model_a

    ```

    """
    src: nn.Module
    dest: nn.Module
    verbose: int = 0

    def __call__(self, x: torch.tensor):
        """Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input.
        Under the hood we tracked all the operations in booth modules.

        :param x: [The input to the modules]
        :type x: torch.tensor
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized

        if len(dest_traced) != len(src_traced):
            raise Exception(
                f'Numbers of operations are different. Source module has {len(src_traced)} operations while destination module has {len(dest_traced)}.')

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f'Transfered from={src_m} to={dest_m}')
            for key in src_m.state_dict().keys():
                assert dest_m.state_dict()[key].sum() == src_m.state_dict()[key].sum()
