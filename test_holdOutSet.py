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

import utils
import dataset
import network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running on device: ', device)

# path to data
main_dir = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/Prostate_QC'
validation_dir = main_dir + '/ValidationSet'
data_csv = validation_dir + '/dataset_information/All_labels.csv'
hdf5_dir = validation_dir + '/subject_hdf5s'
json_dir = validation_dir + '/subject_jsons'

# model directory and name
model_dir = '/scratch_net/bmicdl04/kender/Projects/ProstateQC/models'
model_name = sys.argv[1]

D = dataset.HoldOutValidationSet(hdf5_dir, json_dir, data_csv)
net = network.Network(3, 1)
net.to(device)

# evaluate on the test set:
net.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pth'),
                               map_location=torch.device('cpu')))
test_predictions = []
test_gt_labels = []
test_acc_loss = 0
for tebatch in D.data_loader_3D():
    tebatch_dwi, tebatch_t2ax, tebatch_label = utils.convert_to_torch(tebatch[0],
                                                                      tebatch[1],
                                                                      tebatch[2],
                                                                      device)
    te_out = net(tebatch_dwi, tebatch_t2ax)[1]
    te_out_max = (te_out.detach().cpu().numpy() > 0.0).astype(np.float)
    te_gt_max = tebatch[2]
    test_acc_loss = test_acc_loss + np.sum(np.abs(te_out_max - te_gt_max))                
    test_gt_labels.append(np.squeeze(te_gt_max))
    test_predictions.append(np.squeeze(te_out.detach().cpu().numpy()))
    
print('Final test classificaiton error: ', test_acc_loss / all_samples.size)
np.savetxt(os.path.join(model_dir, model_name+'_predictions_holdOutSet.txt'), test_predictions)
test_predictions = np.asarray(test_predictions)
prob_predictions = 1 / (1 + np.exp(-test_predictions))
np.savetxt(os.path.join(model_dir, model_name+'_prob_predictions_holdOutSet.txt'), prob_predictions)
np.savetxt(os.path.join(model_dir, model_name+'_labels_holdOutSet.txt'), test_gt_labels)

        
D.close_all()    
