from typing import *
from pydantic import BaseModel


class ModelConfig(BaseModel):
    type:str
    params:Dict[str, Any] = {}


class LossConfig(BaseModel):
    type:str
    params:Dict[str, Any] = {}


class DatasetConfig(BaseModel):
    type:str = ''
    path:Union[str, List[str]] = []
    params:Dict[str, Any] = {}


class MetricConfig(BaseModel):
    metric:str
    mode:str = 'min'

    min_delta:float = 1e-5
    patience:int = 100


class OptimizerConfig(BaseModel):
    type:str
    params:Dict[str, Any] = {}


class BackboneFinetuningConfig(BaseModel):
    unfreeze_epoch:int = 10
    incr_rate:float = 1.2
    initial_lr:float = 1e-4

class WrapperConfig(BaseModel):
    type:str = 'ModelWrapper'
    params:Dict[str, Any] = {}


class TrainingConfig(BaseModel):
    model:ModelConfig
    loss:LossConfig
    optimizer:OptimizerConfig

    train_dataset:DatasetConfig
    val_dataset:DatasetConfig
    test_dataset:DatasetConfig = DatasetConfig()

    batch_size:int = 128
    shuffle:bool = True
    num_workers:int = 8

    wrapper:WrapperConfig = WrapperConfig()

    checkpoint_metrics:List[MetricConfig] = [
        MetricConfig(metric='val_loss')
    ]

    early_stop_metrics:List[MetricConfig] = [
        MetricConfig(metric='val_loss')
    ]

    backbone_finetuning:BackboneFinetuningConfig = BackboneFinetuningConfig()

    n_epochs:int = 1000
    n_gpus:Union[str, int, List[int]] = 1

    log_dir:str = 'logs'
