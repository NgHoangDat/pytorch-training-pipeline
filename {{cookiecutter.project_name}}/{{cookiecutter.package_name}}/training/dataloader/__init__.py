from functools import lru_cache

from lescode.export import export_subclass
from torch.utils.data import Dataset


from .registry import DATASETS


@lru_cache
def __init():
    registry = {}
    export_subclass(Dataset, registry=registry)

    for key, cls in registry.items():
        DATASETS.register(key=key)(cls)

    return True

assert __init(), "Failed to load Datasets"
