import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from img_dataloader import ImageDataloader, ImageDataset


class DiffusionModel(pl.LightningModule):
    def __init__(self, ):
        super().__init__()

        self.save_hyperparameters()

        self.unet = Unet()
        self.diffusion = GaussianDiffusion()

    def forward(self, x):
        return self.diffusion(self.unet(x))
    
    def training_step(self, batch, batch_idx):
        low_res, high_res = batch
        loss = self.diffusion.loss(self.unet(low_res), high_res)
        self.log('train_loss', loss)
        return loss
        
