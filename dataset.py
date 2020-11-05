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

class TrainingSet:
    self.DEBUGGING = False
    self.VERBOSE = False
    def __init__(self, _hdf5_dir, _json_dir, _data_csv):
        self.hdf5_dir = _hdf5_dir
        self.json_dir = _json_dir
        self.data_csv = _data_csv
        self.data_df = pd.read_csv(self.data_csv)
        self.subject_ids = self.data_df['study_id'].values
        self.subject_label = self.data_df['DCE_Final'].values
        
        # code for dividing the data in a stratified fashion
        positive_subjects = self.subject_ids[self.subject_label.astype(np.bool)]
        negative_subjects = self.subject_ids[~self.subject_label.astype(np.bool)]
        num_positive_subjects = positive_subjects.size
        num_negative_subjects = negative_subjects.size
        positive_permuted_rows = np.random.permutation(num_positive_subjects)
        negative_permuted_rows = np.random.permutation(num_negative_subjects)

        # training part - 80%
        positive_training_rows = positive_permuted_rows[:np.int(num_positive_subjects*0.8)]
        negative_training_rows = negative_permuted_rows[:np.int(num_negative_subjects*0.8)]
        self.training_samples = np.concatenate((positive_subjects[positive_training_rows], 
                                                negative_subjects[negative_training_rows]))
        self.training_labels = np.concatenate((np.ones(positive_training_rows.size), 
                                               np.zeros(negative_training_rows.size)))
        training_permuted_rows = np.random.permutation(self.training_samples.size)
        self.training_samples = self.training_samples[training_permuted_rows]
        self.training_labels = self.training_labels[training_permuted_rows]
        
        # validation part - 5%
        positive_validation_rows = positive_permuted_rows[np.int(num_positive_subjects*0.8):
                                                          np.int(num_positive_subjects*0.85)]
        negative_validation_rows = negative_permuted_rows[np.int(num_negative_subjects*0.8):
                                                          np.int(num_negative_subjects*0.85)]
        self.validation_samples = np.concatenate((positive_subjects[positive_validation_rows], 
                                                  negative_subjects[negative_validation_rows]))
        self.validation_labels = np.concatenate((np.ones(positive_validation_rows.size), 
                                                 np.zeros(negative_validation_rows.size)))
        validation_permuted_rows = np.random.permutation(self.validation_samples.size)
        self.validation_samples = self.validation_samples[validation_permuted_rows]
        self.validation_labels = self.validation_labels[validation_permuted_rows]
        
        # test part - 15%
        positive_test_rows = positive_permuted_rows[np.int(num_positive_subjects*0.85):]
        negative_test_rows = negative_permuted_rows[np.int(num_negative_subjects*0.85):]
        self.test_samples = np.concatenate((positive_subjects[positive_test_rows], 
                                            negative_subjects[negative_test_rows]))
        self.test_labels = np.concatenate((np.ones(positive_test_rows.size), 
                                           np.zeros(negative_test_rows.size)))
        test_permuted_rows = np.random.permutation(self.test_samples.size)
        self.test_samples = self.test_samples[test_permuted_rows]
        self.test_labels = self.test_labels[test_permuted_rows]
        
        if self.DEBUGGING:
            print('size of training, test and validation parts: ',
                  self.training_samples.size,
                  self.test_samples.size,
                  self.validation_samples.size)
            print('number of positive samples:')
            print('\t in training: ', self.training_labels.sum())
            print('\t in test: ', self.test_labels.sum())
            print('\t in validation: ', self.validation_labels.sum())
            print('intersection between validation and training data: ', 
                  np.intersect1d(self.validation_samples, self.training_samples))
            print('intersection between validation and test data: ', 
                  np.intersect1d(self.validation_samples, self.test_samples))
            print('intersection between test and training data: ', 
                  np.intersect1d(self.test_samples, self.training_samples))
            sys.exit(1)

        if self.VERBOSE: 
            print('Number of positive training examples: ', self.training_labels.sum())
            print('Number of negative training examples: ', (1-self.training_labels).sum())

        
        # getting the training data as hdf5s
        self.training_hdf5s = []
        for tr_sample in self.training_samples: 
            self.training_hdf5s.append(h5py.File(hdf5_dir + '/' + tr_sample + '.hdf5', 'r'))
                
        self.validation_hdf5s = []
        for vl_sample in self.validation_samples: 
            self.validation_hdf5s.append(h5py.File(hdf5_dir + '/' + vl_sample + '.hdf5', 'r'))
            
        self.test_hdf5s = []
        for te_sample in self.test_samples:
            self.test_hdf5s.append(h5py.File(hdf5_dir + '/' + te_sample + '.hdf5', 'r'))

    def training_data_loader_3D(self, batch_size=1): 
        for j in range(self.training_samples.size): 
            dwi = self.training_hdf5s[j]['dwi1'][:]
            dwi = utils.min_max_normalize(dwi)
            dwi = np.expand_dims(dwi.transpose(0,3,1,2), axis=0)
            t2ax = self.training_hdf5s[j]['t2ax_cropped'][:]        
            t2ax = utils.min_max_normalize(t2ax)
            t2ax = t2ax.transpose(2,0,1)
            t2ax = np.expand_dims(np.expand_dims(t2ax, axis=0), axis=0)
            label = np.expand_dims(np.expand_dims(self.training_labels[j], axis=0), axis=0)
            yield dwi, t2ax, label

    def validation_data_loader_3D(self, batch_size=1): 
        for j in range(self.validation_samples.size): 
            dwi = self.validation_hdf5s[j]['dwi1'][:]
            dwi = utils.min_max_normalize(dwi)
            dwi = np.expand_dims(dwi.transpose(0,3,1,2), axis=0)
            t2ax = self.validation_hdf5s[j]['t2ax_cropped'][:]
            t2ax = utils.min_max_normalize(t2ax)
            t2ax = t2ax.transpose(2,0,1)
            t2ax = np.expand_dims(np.expand_dims(t2ax, axis=0), axis=0)
            label = np.expand_dims(np.expand_dims(self.validation_labels[j], axis=0), axis=0)
            yield dwi, t2ax, label

    def test_data_loader_3D(self, batch_size=1): 
        for j in range(self.test_samples.size): 
            dwi = self.test_hdf5s[j]['dwi1'][:]
            dwi = utils.min_max_normalize(dwi)
            dwi = np.expand_dims(dwi.transpose(0,3,1,2), axis=0)
            t2ax = self.test_hdf5s[j]['t2ax_cropped'][:]
            t2ax = utils.min_max_normalize(t2ax)
            t2ax = t2ax.transpose(2,0,1)
            t2ax = np.expand_dims(np.expand_dims(t2ax, axis=0), axis=0)
            label = np.expand_dims(np.expand_dims(self.test_labels[j], axis=0), axis=0)
            yield dwi, t2ax, label

    def close_all(self):
        for f in self.validation_hdf5s:
            f.close()
        for f in self.training_hdf5s:
            f.close()
        for f in test_hdf5s:
            f.close()


class HoldOutValidationSet:
    self.DEBUGGING = False
    self.VERBOSE = False
    def __init__(self, _hdf5_dir, _json_dir, _data_csv):
        self.hdf5_dir = _hdf5_dir
        self.json_dir = _json_dir
        self.data_csv = _data_csv
        self.data_df = pd.read_csv(self.data_csv)
        self.subject_ids = self.data_df['UUID'].values
        self.subject_label = self.data_df['DCE_Final'].values
        self.all_samples = np.asarray(self.subject_ids)
        # getting the data as hdf5s

        self.sample_hdf5s = []
        for sample in self.all_samples:
            self.sample_hdf5s.append(h5py.File(hdf5_dir + '/' + sample + '.hdf5', 'r'))

    def data_loader_3D(self, batch_size=1): 
        for j in range(self.all_samples.size):
            print(self.subject_ids[j])
            try: 
                dwi = self.sample_hdf5s[j]['dwi1'][:]
            except:
                print('dwi1 for {} does not exist'.format(self.subject_ids[j]))
                sys.exit(1)
            dwi = utils.min_max_normalize(dwi)
            dwi = np.expand_dims(dwi.transpose(0,3,1,2), axis=0)
            t2ax = self.sample_hdf5s[j]['t2ax_cropped'][:]
            t2ax = utils.min_max_normalize(t2ax)
            t2ax = t2ax.transpose(2,0,1)
            t2ax = np.expand_dims(np.expand_dims(t2ax, axis=0), axis=0)
            label = np.expand_dims(np.expand_dims(self.test_labels[j], axis=0), axis=0)
            yield dwi, t2ax, label

    def close_all(self):
        for f in self.sample_hdf5s:
            f.close()
