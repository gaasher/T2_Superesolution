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
from utils import ImagePool, set_requires_grad
import wandb

from img_dataloader import ImageDataloader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from models.GAN_modules import SRResNet
import itertools


'''
Implemented based on: https://github.com/deepakhr1999/cyclegans/
'''
class CycleGan(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # generator pair
        self.genX = get_generator()
        self.genY = get_generator()

        
        # discriminator pair
        self.disX = get_disc_model()
        self.disY = get_disc_model()
        
        self.lm = 10.0
        self.idtw = 5.0
        self.sigma = 1.0


        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()
        self.genLoss = None
        self.disLoss = None
    
        #losses
        self.identity_loss = nn.L1Loss()
        self.cycle_loss = nn.L1Loss()
        self.gan_loss = nn.MSELoss()


    def configure_optimizers(self):
        max_lr = 2e-4
        optG = torch.optim.Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=2e-4, betas=(0.5, 0.999))        
        optD = torch.optim.Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters()),
            lr=2e-4, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = torch.optim.lr_scheduler.LambdaLR(optG, lr_lambda=gamma)
        schD = torch.optim.lr_scheduler.LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return self.gan_loss(predictions, target)
            
    def generator_training_step(self, imgA, imgB, stage, batch_idx=1):        
        """cycle images - using only generator nets"""
        fakeB = self.genX(imgA)
        cycledA = self.genY(fakeB)
        
        fakeA = self.genY(imgB)
        cycledB = self.genX(fakeA)
        
        sameB = self.genX(imgB)
        sameA = self.genY(imgA)
        
        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, 'real')
        self.log(f'{stage}_mseGenB', mseGenB)

        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA)
        mseGenA = self.get_mse_loss(predFakeA, 'real')
        self.log(f'{stage}_mseGenA', mseGenA)

        # compute extra losses
        identityLoss = self.idtw*(self.identity_loss(sameA, imgA) + self.identity_loss(sameB, imgB))
        self.log(f'{stage}_identity_loss', identityLoss)

        # compute cycleLosses
        cycleLoss = self.lm*(self.cycle_loss(cycledA, imgA) + self.cycle_loss(cycledB, imgB))
        self.log(f'{stage}_cycle_loss', cycleLoss)

        # gather all losses
        self.genLoss = mseGenA + mseGenB + identityLoss + cycleLoss
        self.log(f'{stage}_gen_total_loss', self.genLoss)
        
        wandb_logger = self.logger.experiment

        if stage == 'val':
            if batch_idx % 100 == 0:
                wandb_logger.log({f'{stage}_realA': wandb.Image(imgA[0], caption='realA')})
                wandb_logger.log({f'{stage}_fakeA': wandb.Image(fakeA[0], caption='fakeA')})
                wandb_logger.log({f'{stage}_realB': wandb.Image(imgB[0], caption='realB')})
                wandb_logger.log({f'{stage}_fakeB': wandb.Image(fakeB[0], caption='fakeB')})


        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()
        
        return self.genLoss
    
    def discriminator_training_step(self, imgA, imgB, stage):
        """Update Discriminator"""        
        fakeA = self.fakePoolA.query(self.fakeA)
        fakeB = self.fakePoolB.query(self.fakeB)
        
        # disX checks for domain A photos
        predRealA = self.disX(imgA)
        mseRealA = self.get_mse_loss(predRealA, 'real')
        self.log(f'{stage}_mseRealA', mseRealA)

        predFakeA = self.disX(fakeA)
        mseFakeA = self.get_mse_loss(predFakeA, 'fake')
        self.log(f'{stage}_mseFakeA', mseFakeA)

        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, 'real')
        self.log(f'{stage}_mseRealB', mseRealB)

        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeB, 'fake')
        self.log(f'{stage}_mseFakeB', mseFakeB)

        # gather all losses
        self.disLoss = 0.1 * (mseFakeA + mseRealA + mseFakeB + mseRealB)
        self.log(f'{stage}_dis_total_loss', self.disLoss)
        return self.disLoss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        imgA, imgB = batch
        discriminator_requires_grad = (optimizer_idx==1)
        set_requires_grad([self.disX, self.disY], discriminator_requires_grad)
        
        if optimizer_idx == 0:
            return self.generator_training_step(imgA, imgB, 'train')
        else:
            return self.discriminator_training_step(imgA, imgB, 'train')        

    def validation_step(self, batch, batch_idx):
        imgA, imgB = batch
        
        loss = self.generator_training_step(imgA, imgB, 'val', batch_idx)
        loss += self.discriminator_training_step(imgA, imgB, 'val')
        return loss
    
if __name__ == '__main__':

    #pl.seed_everything(42)

    wandb_logger = WandbLogger(
    project="SRCycleGAN",
    config="./configs/config-defaults.yaml",
    log_model=True,
    mode="online",
    )

    config = wandb_logger.experiment.config
    wandb_logger.experiment.log_code(".")
    model_name = wandb_logger.experiment.name

    dataset = ImageDataloader(dataset_path='./split_datasets/', batch_size = config['batch_size'], num_workers=8, transforms=True)

    model = CycleGan()
    
   # os.makedirs(f'./trained_models/{model_name}', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'trained_models/{model_name}', monitor='val_gen_total_loss', mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        max_epochs=config['epochs'],
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=.1,
        logger=wandb_logger,
    )

    trainer.fit(model, dataset)