import json
import os
from functools import lru_cache

from class_registry import ClassRegistry
from lescode.export import export_subclass

from .base import Model, ModelConfig

__all__ = ["model_registry", "load_from_dir", "Model", "ModelConfig"]

model_registry = ClassRegistry(unique=True)


def load_from_dir(dir_path: str):
    with open(os.path.join(dir_path, "config.json")) as f:
        cfg = json.load(f)
        model_type = cfg["__type__"]
        model_cls = model_registry.get_class(model_type)
        return model_cls.from_dir(dir_path)


@lru_cache
def __init():
    registry = {}
    export_subclass(Model, registry=registry)

    for key, cls in registry.items():
        model_registry.register(key=key)(cls)

    return True


assert __init(), "Failed to export all Model and ModelConfig"
