from typing import *
from typer import Typer


class Command:

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)


class App:
    
    def __init__(self, *args, **kwargs) -> None:
        self.app = Typer(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self.app(*args, **kwargs)

    def add_typer(self, *args, **kwargs):
        self.app.add_typer(*args, **kwargs)

    def callback(self, *args, **kwargs) -> Any:
        return self.app.callback(*args, **kwargs)

    def command(self, *args, **kwargs):
        def decorator(func:Callable):
            cls = type("Command", Command, {
                "__func": func
            })
            self.app.command(*args, **kwargs)(cls)
            return cls

        return decorator

app = App()
