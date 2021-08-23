from functools import lru_cache

from lescode.export import export_instance

from .app import Command, app

__all__ = ["app"]


@lru_cache
def __init():
    export_instance(Command)
    return True


assert __init(), "Failed to export Command"
