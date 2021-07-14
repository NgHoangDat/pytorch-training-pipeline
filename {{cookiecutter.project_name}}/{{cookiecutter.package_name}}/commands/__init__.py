from functools import lru_cache

from lescode.export import export_subclass

from .app import Command, app


@lru_cache
def __init():
    export_subclass(Command)
    return True

assert __init(), "Failed to export Command"
