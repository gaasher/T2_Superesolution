import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from utils import Compose, RandomCrop, Normalize
from PIL import Image
import torch
import zarr
import pytorch_lightning as pl

class ImageDataset(Dataset):
    def __init__(self, image_dir, stage = 'train', transform=False):
        self.image_dir = image_dir
        self.transform = transform
        self.stage = stage
        self.df = pd.read_pickle(self.image_dir + f'{stage}_df.pkl')

        #self.random_crop = RandomCrop(128)
        #self.normalize = Normalize(mean=0.028408, std=0.0461187)
        #self.transforms = Compose([self.normalize])

    def __len__(self):
        return len(self.df)

    def norm(self, x, max=3729.9300000000076, min=0.0):
        return (x - min) / (max - min)

    def __getitem__(self, idx):
        pt = self.df.iloc[idx]['pt']
        slice_num = self.df.iloc[idx]['slice_num']

        low_res = zarr.load(f'{self.image_dir}/{self.stage}/low_res_{pt}_{slice_num}.zarr')
        high_res = zarr.load(f'{self.image_dir}/{self.stage}/hi_res_{pt}_{slice_num}.zarr')
        
        #print(low_res.shape, high_res.shape)

        low_res = self.norm(low_res)
        high_res = self.norm(high_res)


        # low_res = Image.fromarray(low_res)
        # high_res = Image.fromarray(high_res)

        # if self.transform:
        #     low_res, high_res = self.transforms(low_res, high_res)
        
        low_res = torch.from_numpy(np.array(low_res)).float()
        high_res = torch.from_numpy(np.array(high_res)).float()


        low_res = low_res.unsqueeze(0)
        high_res = high_res.unsqueeze(0)
        
        return low_res, high_res
    

class ImageDataloader(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size,
                 num_workers,
                 transforms=False,
                 pin_memory=True,
                 shuffle=True):
        super().__init__()

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self.train_dataset = ImageDataset(self.dataset_path, stage='train', transform=self.transforms)
        self.val_dataset = ImageDataset(self.dataset_path, stage='val', transform=self.transforms)
        self.test_dataset = ImageDataset(self.dataset_path, stage='test', transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=self.shuffle)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)