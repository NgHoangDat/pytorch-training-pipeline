from typing import *
from pathlib import Path
from abc import ABCMeta, abstractmethod, abstractclassmethod, abstractstaticmethod

import torch
from torch import nn
from pydantic import BaseModel


class ModelConfig(BaseModel):
    pass


class Model(nn.Module, metaclass=ABCMeta):

    @abstractstaticmethod
    def get_config_cls() -> Type[ModelConfig]:
        pass

    @abstractclassmethod
    def from_dir(cls, model_dir:str, *args, **kwargs):
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @classmethod
    def from_dir(cls, model_dir: str, *args, cfg:str='config.json', weight:str='weight.pth', **kwargs):
        model_dir = Path(model_dir).resolve()
        cfg = cls.get_config_cls().parse_file(model_dir.joinpath(cfg).as_posix())
        model = cls(cfg, False)
                
        if weight:
            weight = torch.load(model_dir.joinpath(weight).as_posix())
            model.load_state_dict(weight, strict=False)

        return model
