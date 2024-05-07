import glob
import os
from typing import List, Tuple, Optional, Dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from omegaconf import DictConfig
from flwr.server.history import History
from dataset import *
from torch.utils.data import Sampler, RandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.transforms import ToTensor, Compose

from torchvision.transforms import ToTensor, Normalize, Compose, RandomResizedCrop, RandomApply, ColorJitter, RandomGrayscale, RandomHorizontalFlip, Resize
import torchvision.transforms.functional as F
# from .functional import F
# def resize_img(img):
#     return F.resize(img, size=224, interpolation=F.InterpolationMode.BILINEAR) #BICUBIC
# 224 , 256, 384
class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""

        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            return [q, k]
        
def main():
    datapath = 'D:/Datasets/data' #'D:/DesktopC/Datasets/data/' 
    subset = True #True
    num_classes = 10
    num_workers = 1
    batch_size=32
    seed=2024
    num_partitions = 10
    alpha = 0.5
    partitioning = 'dirichlet' 
    balance=True
    # partitioning = 'iid'
    val_ratio = 0.3

    augmentation = Compose([
            RandomResizedCrop(384, scale=(0.4, 1.), interpolation= F.InterpolationMode.BICUBIC),
            RandomApply([
                ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            RandomGrayscale(p=0.2),
            RandomHorizontalFlip(),
            ToTensor(),
            # normalize
        ])
    simple_trf = Compose([
        # Resize(224, F.InterpolationMode.BICUBIC),
        Resize((384, 384), interpolation= F.InterpolationMode.BICUBIC),
        ToTensor(),
        # resize_img, # avoid pickling error
    ])
    # F.resize(image, [224, 224])

    trainset = Food101(root=datapath, split="train", transform=TwoCropsTransform(augmentation), download= True)
    testset = Food101(root=datapath, split="test", transform=simple_trf, download= True)

    # print(type(testset))
    if subset:
            #Taking Subset of trainset and testset
            # select classes you want to include in your subset
            list = [i for i in range(num_classes)]
            classes = torch.tensor(list)
            # get indices that correspond to one of the selected classes
            train_indices = (torch.tensor(trainset._labels)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
            np_tr_idx = np.array(train_indices)
            # subset the dataset
            train_sub = CustomSubset(trainset, np_tr_idx) # tr_mapped_lab)
            # get indices that correspond to one of the selected classes
            test_indices = (torch.tensor(testset._labels)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
            np_test_idx = np.array(test_indices)
            # subset the dataset
            test_sub = CustomSubset(testset, np_test_idx) #, test_mapped_lab)
            # test_sub = Subset(testset, test_indices)
        #     return train_sub, test_sub
        # else:
        #     return trainset, testset

    print(type(train_sub))
    print(train_sub.dataset)

    trainset = train_sub
    testset = test_sub
    train_loader = DataLoader(Subset(trainset.dataset, trainset.indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(Subset(testset.dataset, testset.indices), batch_size=batch_size, num_workers=num_workers)

    print(type(test_loader), type(train_loader))
    for X, y in test_loader:
        print(X.size())
        break
    
    num_samples_train = len(trainset)
    num_loads_train = len(train_loader)
    # num_batches_train = num_samples_train // bs
    print(num_samples_train, num_loads_train)
    for data in train_loader:
         print(type(data), len(data))
         print(type(data[0]))
         print(data[0][0].size())
         print(data[0][1].size())
               #, type(data[0]), type(data[1]), type(data[2]))
         break
if __name__ == "__main__":
    main()
