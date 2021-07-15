import json
import os.path as op
import sys
import warnings
from pathlib import Path
from typing import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BackboneFinetuning
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from typer import Argument, Option

sys.path.append(op.abspath(op.join(__file__, op.pardir, op.pardir)))

from .app import app

try:
    from core import *
    from training import *
except ImportError:
    from ..core import *
    from ..training import *

warnings.filterwarnings('once')


@app.command()
def train(
        model_dir:str = Argument(..., help='Model directory'),
        log_dir:str = Option("logs", help="Log directory"),
        config_path:str=Option("", help='Path to config file')
    ):
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(log_dir).joinpath(model_dir.stem).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if not config_path:
        config_path = model_dir.joinpath('config.json').as_posix()

    config = TrainingConfig.parse_file(config_path)

    model_cls:Model = get_model_cls(config.model.type)
    if model_cls is None:
        raise ValueError(f"Model {config.model.type} is not supported")

    model_cfg:ModelConfig = model_cls.get_config_cls()(**config.model.params)
    print(model_cfg)
    model = model_cls(model_cfg)
    
    train_dataset = DATASETS.get(config.train_dataset.type, 
        **config.train_dataset.params
    )

    val_dataset = DATASETS.get(config.val_dataset.type,
        **config.val_dataset.params
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)
    val_dataloader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    with open(model_dir.joinpath('config.json'), 'w') as f:
        config_data = config.dict()
        config_data['model']['params'] = model_cfg.dict()
        json.dump(config_data, f, indent=4, sort_keys=True)

    checkpoint_callbacks = [
        ModelCheckpoint(
            dirpath=model_dir.as_posix(),
            save_last=True,
            save_top_k=1,
            verbose=True,
            monitor=cfg.metric,
            mode=cfg.mode,
            filename=f"best_{cfg.metric}"
        )
        for cfg in config.checkpoint_metrics
    ]

    early_stopping_callbacks = [
        EarlyStopping(
            monitor=cfg.metric,
            min_delta=cfg.min_delta,
            patience=cfg.patience
        ) for cfg in config.early_stop_metrics
    ]

    loss = LOSSES.get(config.loss.type, n_classes=model_cfg.get_num_classes(), **config.loss.params)
    optimizer = OPTIMIZERS.get(config.optimizer.type, params=model.parameters(), **config.optimizer.params)

    checkpoint_path = model_dir.joinpath('last.ckpt')
    wrapper_cls = get_wrapper_cls(config.wrapper.type)

    if checkpoint_path.exists():
        wrapper = wrapper_cls.load_from_checkpoint(
            checkpoint_path=checkpoint_path.as_posix(), 
            model=model, loss=loss, optimizer=optimizer,
            **config.wrapper.params
        )
    else:
        wrapper = wrapper_cls(model, loss=loss, optimizer=optimizer, **config.wrapper.params)

    trainer = pl.Trainer(
        callbacks=[
            BackboneFinetuning(
                unfreeze_backbone_at_epoch=config.backbone_finetuning.unfreeze_epoch,
                lambda_func=lambda _: config.backbone_finetuning.incr_rate,
                backbone_initial_lr=config.backbone_finetuning.initial_lr,
                should_align=True
            )
        ] + early_stopping_callbacks + checkpoint_callbacks, 
        max_epochs=config.n_epochs, gpus=config.n_gpus,
        logger=TensorBoardLogger(log_dir.as_posix())
    )

    trainer.fit(wrapper, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
