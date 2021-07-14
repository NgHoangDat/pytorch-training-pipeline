from functools import wraps, lru_cache
from typing import *
from typer import Typer

__all__ = ["app", "Command"]


@lru_cache
def get_mock_typer():
    return Typer()


class Command:
    call:Callable = lambda *args, **kwargs: None


class App(Typer):
    
    @wraps(get_mock_typer().command)
    def command(self, *args, **kwargs):
        base = super()

        def decorator(func:Callable):
            base.command(*args, **kwargs)(func)
            cls = type("Command", (Command,), {
                "call": func
            })
            return cls

        return decorator

app = App()
