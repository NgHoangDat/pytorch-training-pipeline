from functools import lru_cache

from lescode.export import export_subclass
from torch import optim
from torch.optim import Optimizer

from .registry import OPTIMIZERS


@lru_cache
def __init():
    registry = {}
    
    export_subclass(Optimizer, module=optim, registry=registry)
    export_subclass(Optimizer, registry=registry)

    for key, cls in registry.items():
        OPTIMIZERS.register(key=key)(cls)

    return True


assert __init(), "Failed to load Optimizers"
