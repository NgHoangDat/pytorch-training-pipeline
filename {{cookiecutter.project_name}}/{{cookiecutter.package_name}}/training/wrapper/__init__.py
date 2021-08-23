from functools import lru_cache

from class_registry import ClassRegistry
from lescode.export import export_subclass

from .base_wrapper import ModelWrapper

__all__ = ["wrapper_registry", "ModelWrapper"]

wrapper_registry = ClassRegistry(unique=True)


@lru_cache
def __init():
    registry = {}
    export_subclass(ModelWrapper, registry=registry)

    for key, cls in registry.items():
        wrapper_registry.register(key=key)(cls)

    return True


assert __init(), "Failed to load wrapper"
