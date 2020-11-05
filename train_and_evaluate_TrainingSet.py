import numpy as np
import json
from glob import glob
import h5py
import pandas as pd
from scipy.ndimage import zoom as imresize
import sys
import os.path
from scipy.linalg import inv

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
hdf5_dir = main_dir + '/subject_hdf5s'
json_dir = main_dir + '/subject_jsons'
data_csv = main_dir + '/dataset_information/All_labels_reduced.csv'

# model directory and name
model_dir = '/scratch_net/bmicdl04/kender/Projects/ProstateQC/models'
model_name = sys.argv[1]

D = dataset.TrainingSet(hdf5_dir, json_dir, data_csv)
net = network.Network(3, 1)
net.to(device)

# Train on training partition for 30 epochs
optimizer = optim.Adam(net.parameters(), lr=0.00005, weight_decay=0.0)
net.zero_grad()
loss_function = torch.nn.BCEWithLogitsLoss()
bsize, n = 5, 0
avg_loss = 0
min_vl_acc_loss = 1.0
min_vl_loss = 1.0
vl_acc_loss_buffer = []
for epoch in range(30): 
    for batch in D.training_data_loader_3D():
        batch_dwi = batch[0]
        batch_t2 = batch[1]
        batch_label = batch[2]
        batch_dwi, batch_t2 = utils.flip_transform(batch_dwi, batch_t2, 0.5)
        batch_dwi, batch_t2, batch_label = utils.convert_to_torch(batch_dwi,
                                                                  batch_t2,
                                                                  batch_label,
                                                                  device)
        out = net(batch_dwi, batch_t2)
        loss = loss_function(out[1], batch_label)/bsize
        avg_loss = avg_loss + loss.detach().cpu().numpy()                
        loss.backward()        
        n += 1
        if n == bsize:
            # Compute validation loss
            vl_acc_loss = 0            
            vl_loss = 0
            for vlbatch in D.validation_data_loader_3D():
                vlbatch_dwi, vlbatch_t2ax, vlbatch_label = utils.convert_to_torch(vlbatch[0],
                                                                                  vlbatch[1],
                                                                                  vlbatch[2],
                                                                                  device)
                vl_out = net(vlbatch_dwi, vlbatch_t2ax)[1]
                vl_out_max = (vl_out.detach().cpu().numpy() > 0.0).astype(np.float)
                vl_gt_max = vlbatch[2]
                vl_acc_loss = vl_acc_loss + np.sum(np.abs(vl_out_max - vl_gt_max))                
                vl_loss = vl_loss + loss_function(vl_out, vlbatch_label).detach().cpu().numpy()
                
            vl_loss = vl_loss / validation_samples.size
            vl_acc_loss = vl_acc_loss / validation_samples.size
            vl_acc_loss_buffer.append(vl_acc_loss)            
            average_size = np.min([5, len(vl_acc_loss_buffer)])
            last_five_average = np.mean(vl_acc_loss_buffer[-average_size])
            print('epoch: {} >> val. classification error: {}'.format(epoch, vl_acc_loss))
            print('epoch: {} >> val. classification error average of last five: {}'.format(epoch, last_five_average))
            print('epoch: {} >> val. loss: {}'.format(epoch, vl_loss))
            print('epoch: {} >> training avg loss: {}'.format(epoch, avg_loss))
            # If the validation error reaches a running average minimum save model
            if last_five_average < min_vl_acc_loss:
                print('new minimum val. cl. error: {} (old loss: {})'.format(last_five_average, min_vl_acc_loss))
                torch.save(net.state_dict(), os.path.join(model_dir, model_name+'.pth'))
                min_vl_acc_loss = last_five_average
                min_vl_loss = vl_loss
            elif (last_five_average == min_vl_acc_loss) and (vl_loss < min_vl_loss):
                print('new minimum val. loss at the same error rate: {} (old loss: {})'.format(vl_loss,
                                                                                               min_vl_loss))
                torch.save(net.state_dict(), os.path.join(model_dir, model_name+'.pth'))
                min_vl_acc_loss = last_five_average
                min_vl_loss = vl_loss                
            # take an optimization step
            optimizer.step()
            optimizer.zero_grad()
            
            n = 0
            avg_loss = 0

# evaluate on the test partition
net.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pth')))
test_acc_loss = 0            
test_loss = 0
test_predictions = []
test_gt_labels = []
for tebatch in D.test_data_loader_3D():
    tebatch_dwi, tebatch_t2ax, tebatch_label = utils.convert_to_torch(tebatch[0],
                                                                      tebatch[1],
                                                                      tebatch[2],
                                                                      device)
    te_out = net(tebatch_dwi, tebatch_t2ax)[1]
    te_out_max = (te_out.detach().cpu().numpy() > 0.0).astype(np.float)
    te_gt_max = tebatch[2]
    test_acc_loss = test_acc_loss + np.sum(np.abs(te_out_max - te_gt_max))                
    test_loss = test_loss + loss_function(te_out, tebatch_label).detach().cpu().numpy()
    test_gt_labels.append(np.squeeze(te_gt_max))
    test_predictions.append(np.squeeze(te_out.detach().cpu().numpy()))
    
print('Final test loss: ', test_loss / test_samples.size)
print('Final test classificaiton error: ', test_acc_loss / test_samples.size)
np.savetxt(os.path.join(model_dir, model_name+'labels.txt'), test_gt_labels)
np.savetxt(os.path.join(model_dir, model_name+'predictions.txt'), test_predictions)
        
# closing hdf5s
D.close_all()
    
