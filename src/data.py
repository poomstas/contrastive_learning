# %%
import sys
sys.path.append('..')

import os
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.datasets import ShapeNet

from src.paths import DATA

# %%
class PointCloudData(Dataset):
    def __init__(self, 
                 n_dataset          = 5000, 
                 categories         = ['Table', 'Lamp', 'Guitar', 'Motorbike']):
        self.data = ShapeNet(root=DATA, categories=categories).shuffle()[:n_dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.train[index]
 

# %% Test for FirstStagetData Dataset Class
if __name__=='__main__':
    data = PointCloudData()
