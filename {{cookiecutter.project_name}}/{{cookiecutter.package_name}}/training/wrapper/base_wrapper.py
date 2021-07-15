from typing import *

import pytorch_lightning as pl
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer


class ModelWrapper(pl.LightningModule):

    def __init__(self, model:nn.Module, *args: Any, loss:nn.Module=None, optimizer:Optimizer=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = getattr(model, 'backbone', None)
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, *args, **kwargs) -> Any:
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch:Tuple[Tensor], *args, **kwargs):
        images, gold = batch

        pred = self.forward(images)
        loss = self.loss(pred, gold)

        self.log("train_loss", loss)
        return loss

    def infer_step(self, phase:str, batch:Tuple[Tensor], *args, **kwargs):
        inputs, gold = batch
        pred = self.forward(inputs)
        
        loss = self.loss(pred, gold)

        output = {
            f'{phase}_loss': loss
        }

        for key, val in output.items():
            self.log(key, val)

        return output

    def validation_step(self, batch:Tuple[Tensor], *args, **kwargs):
        return self.infer_step("val", batch=batch)

    def test_step(self, batch:Tuple[Tensor], *args, **kwargs):
        return self.infer_step("test", batch=batch)

    def configure_optimizers(self):
        return self.optimizer
