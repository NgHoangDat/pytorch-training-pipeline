from functools import lru_cache

from lescode.export import export_subclass

from .dataset import Dataset
from .registry import dataset_registry

__all__ = ["dataset_registry", "Dataset"]


@lru_cache
def __init():
    registry = {}
    export_subclass(Dataset, registry=registry)

    for key, cls in registry.items():
        dataset_registry.register(key=key)(cls)

    return True


assert __init(), "Failed to load Datasets"
