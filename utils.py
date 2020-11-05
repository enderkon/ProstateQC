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

def one_hot_encoding(labels, num_class, starts_from_zero=True): 
    out = np.zeros([labels.shape[0], num_class])
    for j in range(out.shape[0]): 
        if starts_from_zero: 
            out[j, np.int(labels[j])] = 1
        else: 
            out[j, np.int(labels[j])-1] = 1
    return out

def min_max_normalize(I, min_p=20, max_p=80): 
    if I.ndim == 3: 
        mI = np.percentile(I, min_p)
        MI = np.percentile(I, max_p)
        return ((I - mI) / (MI - mI) - 0.5)*5.0
    elif I.ndim == 4: 
        K = np.zeros_like(I)
        for j in range(I.shape[0]):
            K[j] = min_max_normalize(I[j])
        return K

def flip_transform(dwi, t2ax, prob_flip=0.2):
    a = np.random.rand()
    if a < prob_flip:
        return np.flip(dwi, axis=3).copy(), np.flip(t2ax, axis=3).copy()
    else:
        return dwi, t2ax

def convert_to_torch(dwi, t2ax, label, device):
    return torch.from_numpy(dwi).float().to(device), \
        torch.from_numpy(t2ax).float().to(device), \
        torch.from_numpy(label).to(device)
