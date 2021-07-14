import json
import os
from functools import lru_cache

from lescode.export import export_subclass

from .base_model import Model, ModelConfig

__all__ = [
    'get_model_cls', 
    'load_from_dir', 
    'Model', 
    'ModelConfig'
]


@lru_cache
def _get_registry(key:str):
    return {}


def get_model_cls(name:str) -> Model:
    return _get_registry("model").get(name)


def load_from_dir(dir_path:str):
    with open(os.path.join(dir_path, 'config.json')) as f:
        cfg = json.load(f)
        model_type = cfg['__type__']
        model_cls = get_model_cls(model_type)
        return model_cls.from_dir(dir_path)


@lru_cache
def __init():
    export_subclass(Model, registry=_get_registry("model"))
    export_subclass(ModelConfig, registry=_get_registry("model_config"))
    return True

assert __init(), "Failed to export all Model and ModelConfig"
