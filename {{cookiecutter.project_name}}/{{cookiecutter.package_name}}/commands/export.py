import json
import re
import sys
from pathlib import Path
from typing import *

import torch
import torch.onnx
from torch.nn import Module
from typer import Argument, Option

from .app import app

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

try:
    from core import *
    from training import *
except ImportError:
    from ..core import *
    from ..training import *


def save_as_torch(model:Module, target_dir:Path, *args, **kwargs):
    torch.save(model.state_dict(), target_dir.joinpath('weight.pth').as_posix())


def save_as_onnx(
        model:Module, target_dir:Path, cfg:ModelConfig, 
        batch_size:int, prefix:str, *args, **kwargs
    ):
    raise NotImplementedError()


FORMATS = {
    "torch": save_as_torch,
    "onnx": save_as_onnx
}


@app.command()
def export(
        checkpoint_dir:str = Argument(..., help="Checkpoint directory"),
        target_dir:str = Argument(..., help="Target directory"),
        
        format:str = Option('torch', help="Export format"),

        batch_size:int = Option(4, help="Batch size"),

        variant:str = Option('best_val_loss', help='Checkpoint variant'),
        version:str = Option('', help="Checkpoint version"),
    ):

    checkpoint_dir = Path(checkpoint_dir).resolve()

    if format not in FORMATS:
        raise ValueError(f"{format} is not supported")

    if not version:
        variants = [
            fn.stem for fn in checkpoint_dir.rglob(f'{variant}*.ckpt')
        ]
        variants.sort()
        match = re.search(f'(?<={variant}).+', variants[-1])
        if match:
            version = match.group()
    
    checkpoint_path = checkpoint_dir.joinpath(f"{variant}{version}.ckpt").as_posix()
    config_path = checkpoint_dir.joinpath('config.json').as_posix()

    config = TrainingConfig.parse_file(config_path)
    model_cls:Model = get_model_cls(config.model.type)

    if model_cls is None:
        raise ValueError(f"Model {config.model.type} is not supported")

    model_cfg:ModelConfig = model_cls.get_config_cls()(**config.model.params)
    base = model_cls(model_cfg)

    wrapper = get_wrapper_cls(config.wrapper.type).load_from_checkpoint(
        checkpoint_path=checkpoint_path, model=base, strict=False,
        **config.wrapper.params
    )
    wrapper.eval()
    model = getattr(wrapper, 'model')

    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with open(target_dir.joinpath('config.json'), 'w') as f:
        cfg = model_cfg.dict()
        cfg['__type__'] = config.model.type
        json.dump(cfg, f)

    FORMATS[format](
        model, target_dir=target_dir, 
        cfg=model_cfg, batch_size=batch_size
    )

    print(f'Export model successfully to {target_dir.as_posix()}')
