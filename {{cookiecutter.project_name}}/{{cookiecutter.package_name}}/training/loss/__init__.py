from functools import lru_cache

from lescode.export import export_subclass

from .loss import Loss
from .registry import LOSSES


@lru_cache
def __init():
    registry = {}
    export_subclass(Loss, registry=registry)

    for key, cls in registry.items():
        LOSSES.register(key=key)(cls)

    return True


assert __init(), "Failed to load Losses"
