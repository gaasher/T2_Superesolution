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
from models.GAN_modules import GeneratorResNet, Discriminator
import itertools
from models.GAN_losses import VGGLoss, GANLoss, TVLoss, PSNR, GradLoss
from kornia.losses import ssim
from torchvision import transforms


'''
Implemented based on: https://github.com/deepakhr1999/cyclegans/
'''
class SRGAN(pl.LightningModule):
    def __init__(self, num_resnet_blocks, max_lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        hr_shape = (256, 256)

        self.max_lr = max_lr
        # generator pair
        self.genY = GeneratorResNet(n_residual_blocks=9)
        
        # discriminator pair
        self.disY = Discriminator(input_shape=(1, *hr_shape))
    
        #set criterios
        self.criterion_MSE = nn.MSELoss()
        self.criterion_MAE = nn.L1Loss()
        #self.criterion_VGG = VGGLoss(num_channels=1)
        self.criterion_GAN = GANLoss(gan_mode='lsgan')
        self.criterion_TV = TVLoss()
        self.grad_loss = GradLoss()

        # self.unnormalize = transforms.Compose([ transforms.Normalize(mean = [0.],
        #                                              std = [ 1/0.028408,]),
        #                         transforms.Normalize(mean = [-0.046118],
        #                                              std = [ 1.]),
        #                        ])
    
    def renormalize(self, img):
        return img * 3729.9300000000076

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgA, imgB = batch
        img_gen = self.genY(imgA)

        if optimizer_idx == 0:
            return self.generator_training_step(img_gen, imgB, 'train', batch_idx)
        else:
            return self.discriminator_training_step(img_gen, imgB, 'train')        

    def validation_step(self, batch, batch_idx):
        imgA, imgB = batch
        img_gen = self.genY(imgA)

        loss = self.generator_training_step(img_gen, imgB, 'val', batch_idx)
        loss += self.discriminator_training_step(img_gen, imgB, 'val')
        return loss

    def discriminator_training_step(self, imgA, imgB, mode):
        #real
        d_out_real = self.disY(imgB)
        d_loss_real = self.criterion_GAN(d_out_real, True)
        self.log(f'{mode}_dis_real_loss', d_loss_real)
        #fake
        d_out_fake = self.disY(imgA)
        d_loss_fake = self.criterion_GAN(d_out_fake, False)
        self.log(f'{mode}_dis_fake_loss', d_loss_fake)

        #combined discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        self.log(f'{mode}_dis_total_loss', d_loss)
        return d_loss

    def generator_training_step(self, imgA, imgB, mode, batch_idx=None):
        mse_loss = self.criterion_MAE(imgA, imgB)
        #print(mse_loss)
        #vgg_loss = self.criterion_VGG(imgA, imgB)
        #print(vgg_loss)
        content_loss = mse_loss
        self.log(f'{mode}_gen_content_loss', content_loss)
        #adversarial loss
        adv_loss = self.criterion_GAN(self.disY(imgA), True)
        self.log(f'{mode}_gen_adv_loss', adv_loss)
        #tv_loss
        tv_loss = self.criterion_TV(imgA)
        self.log(f'{mode}_gen_tv_loss', tv_loss)
        #grad loss
        grad_loss = self.grad_loss(imgA, imgB)
        self.log(f'{mode}_gen_grad_loss', grad_loss)

        combined_loss = 2.0 * content_loss + 1e-3 * adv_loss + 1e-8 * tv_loss + 1e-5 * grad_loss
        self.log(f'{mode}_gen_total_loss', combined_loss)
    
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        #mae
        # imgA = self.unnormalize(imgA)
        # imgB = self.unnormalize(imgB)
        # imgA = self.renormalize(imgA)
        # imgB = self.renormalize(imgB)
        mae = torch.mean(torch.abs(self.renormalize(imgA) - self.renormalize(imgB)))
        self.log(f'{mode}_gen_mae', mae)

        wandb_logger = self.logger.experiment

        if mode == 'train':
            if batch_idx % 500 == 0:
                    wandb_logger.log({f'{mode}_fakeHiRes': wandb.Image(imgA[0], caption='FaleHiRes')})
                    wandb_logger.log({f'{mode}_realHiRes': wandb.Image(imgB[0], caption='RealHiRes')})

        if mode == 'val':
                if batch_idx % 250 == 0:
                    wandb_logger.log({f'{mode}_fakeHiRes': wandb.Image(imgA[0], caption='fakeHiRes')})
                    wandb_logger.log({f'{mode}_realHiRes': wandb.Image(imgB[0], caption='realHiRes')})

        return combined_loss

    def configure_optimizers(self):
        optG = torch.optim.Adam(self.genY.parameters(), lr=self.max_lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(self.disY.parameters(), lr=self.max_lr, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        # schG = torch.optim.lr_scheduler.LambdaLR(optG, lr_lambda=gamma)
        # schD = torch.optim.lr_scheduler.LambdaLR(optD, lr_lambda=gamma)
        schG = torch.optim.lr_scheduler.OneCycleLR(optG, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches)
        schD = torch.optim.lr_scheduler.OneCycleLR(optD, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches)
    
        return [optG, optD], [schG, schD]


if __name__ == '__main__':

    #pl.seed_everything(42)

    wandb_logger = WandbLogger(
    project="SRGAN-L1-GradLoss",
    config="./configs/config-defaults.yaml",
    log_model=True,
    mode="online",
    )

    config = wandb_logger.experiment.config
    wandb_logger.experiment.log_code(".")
    model_name = wandb_logger.experiment.name

    dataset = ImageDataloader(dataset_path='./split_datasets/', batch_size = config['batch_size'], num_workers=8, transforms=True)

    model = SRGAN(num_resnet_blocks=config['num_residual_blocks'])
    
   # os.makedirs(f'./trained_models/{model_name}', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'trained_models/{model_name}', monitor='val_gen_total_loss', mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=32,
        max_epochs=config['epochs'],
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=.5,
        logger=wandb_logger,
    )

    trainer.fit(model, dataset)