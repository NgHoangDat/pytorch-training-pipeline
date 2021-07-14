from functools import lru_cache

from lescode.export import export_subclass

from .base_wrapper import ModelWrapper

__all__ = ['get_wrapper_cls', 'ModelWrapper']


@lru_cache
def __get_wrapper_registry():
    return {}


def get_wrapper_cls(name:str):
    return __get_wrapper_registry().get(name)


@lru_cache
def __init():
    export_subclass(ModelWrapper, registry=__get_wrapper_registry())
    return True

assert __init(), "Failed to load wrapper"
