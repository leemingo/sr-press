"""Implements the SoccerMap architecture."""
from typing import Any, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

class BaseLine(nn.Module):
    def __init__(self, model_config: dict = None):
        super().__init__()

        in_channels = model_config.get('in_channels', 7)
        self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=(5, 5), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding="valid")
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(32, 1, kernel_size=(1, 1))
        
        self.symmetric_padding = nn.ReplicationPad2d((2, 2, 2, 2))
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.output_layer = nn.Linear(1, 1)   

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.symmetric_padding(x)
        x = F.relu(self.conv_2(x))
        x = self.symmetric_padding(x)

        x = F.relu(self.conv3(x))
        x = self.conv4(x) 

        x = self.global_pooling(x).view(x.size(0), -1) 
        x = self.output_layer(x)

        return x
    
class PytorchModel(pl.LightningModule):
    """A pass success probability model based on the SoccerMap architecture."""

    def __init__(
        self,
        model,
        optimizer_params: dict = None
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = model
        self.sigmoid = nn.Sigmoid()

        # loss function
        self.criterion = torch.nn.BCELoss()
        self.optimizer_params = optimizer_params

    def forward(self, x: torch.Tensor):
        x = self.model(x)   
        x = self.sigmoid(x)        

        return x

    def step(self, batch: Any):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train_loss", loss, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val_loss", loss, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True,
                 sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch

        preds = self(x) # self.forward(x)

        return preds, y

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return torch.optim.Adam(self.parameters(), **self.optimizer_params["optimizer_params"])


    