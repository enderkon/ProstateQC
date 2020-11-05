import numpy as np
import json
from glob import glob
import h5py
import pandas as pd
from scipy.ndimage import zoom as imresize
import sys
import os.path
from scipy.linalg import inv
import utils

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# network module
class Network(nn.Module):
    def __init__(self, n_in_dwi, n_in_st=1):
        super(Network, self).__init__()
        self.dwi_cnn = nn.Sequential(
            nn.Conv3d(n_in_dwi, 8, 3),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3), 
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 16, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.MaxPool3d([1,2,2], stride=[1,2,2]),
            nn.Conv3d(32, 64, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.MaxPool3d([1,2,2], stride=[1,2,2]),
            nn.Conv3d(32, 64, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, padding=[1,0,0])
        )
        self.t2ax_cnn = nn.Sequential(
            nn.Conv3d(n_in_st, 8, 3),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3), 
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 16, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.MaxPool3d([1,2,2], stride=[1,2,2]),
            nn.Conv3d(32, 64, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, padding=[1,0,0]), 
            nn.ReLU(),
            nn.MaxPool3d([1,2,2], stride=[1,2,2]),
            nn.Conv3d(32, 64, 3, padding=[1,0,0]), 
            nn.ReLU(),            
            nn.Conv3d(64, 32, 3, padding=[1,0,0])
        )
        self.dnn = nn.Sequential(
            nn.Linear(64, 1),
        )
    def forward(self, dwi, t2ax): 
        out_dwi = self.dwi_cnn(dwi)
        out_t2ax = self.t2ax_cnn(t2ax)
        summary_dwi = out_dwi.mean([2,3,4])
        summary_t2ax = out_t2ax.mean([2,3,4])
        summary = torch.cat((summary_dwi, summary_t2ax), axis=1)
        out = self.dnn(summary)
        return F.softmax(out, dim=1), out
