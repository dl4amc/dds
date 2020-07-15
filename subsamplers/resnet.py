#!/usr/bin/env python
# coding: utf-8

# Import required modules
from keras import layers
from keras import models
import os, random, keras, cPickle
# import _pickle as cPickle
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from copy import deepcopy
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import AlphaDropout
from keras.optimizers import adam
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
K.tensorflow_backend._get_available_gpus()
K.set_image_dim_ordering('th')


# data pre-processing
Xd = cPickle.load(open("../data/RML2016.10b_dict.dat", 'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
print("length of snr",len(snrs))
print("length of mods",len(mods))
X = [] 
Y_snr = []
lbl = []
for snr in snrs:
    for mod in mods:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod, snr))
        Y_snr = Y_snr + [mod]*6000
X = np.vstack(X)
Y_snr = np.vstack(Y_snr)
print("shape of X", np.shape(X))


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


# Use only the train split
np.random.seed(2016)
n_examples = X.shape[0]
n_train_valid = n_examples // 2
train_valid_idx = np.random.choice(range(0, n_examples), size=n_train_valid, replace=False)
X_train_valid = X[train_valid_idx]
n_train = 3 * n_train_valid // 4
train_idx = np.random.choice(range(0, n_train_valid), size=n_train, replace=False)
X = X_train_valid[train_idx]
valid_idx = list(set(range(0, n_train_valid))-set(train_idx))
X_valid = X_train_valid[valid_idx]
Y_snr = to_onehot(map(lambda x: mods.index(lbl[x][0]), range(X.shape[0])))

print("shape of X", np.shape(X))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
num_samples = 64
new_X = []
orig_model = load_model('../models/resnet_ranker.h5')
for eva_iter in range(X.shape[0]//60000):
    snr_data = X[eva_iter*60000:(eva_iter+1)*60000]
    snr_out = Y_snr[eva_iter*60000:(eva_iter+1)*60000]
    snr_acc_list = []
    snr_data_copy = deepcopy(snr_data)
    for idx in range(X.shape[2]):
        snr_data = deepcopy(snr_data_copy)
        snr_data = snr_data.transpose((2, 1, 0))
        new_snr_data = np.append(snr_data[:idx], np.zeros((1, snr_data.shape[1], snr_data.shape[2])), axis=0)
        snr_data = np.append(new_snr_data, snr_data[idx+1:], axis=0)
        snr_data = snr_data.transpose((2, 1, 0))
        score = orig_model.evaluate(snr_data, snr_out, batch_size=60000, verbose=0)
        snr_acc_list.append((idx, score[1]))
    snr_acc_list.sort(key=lambda x: x[1])
    snr_acc_list = snr_acc_list[:num_samples]
    snr_acc_list.sort(key=lambda x: x[0]) 
    snr_idxs = [ele[0] for ele in snr_acc_list]
    snr_data = snr_data.transpose((2, 1, 0))
    snr_data = snr_data[snr_idxs]
    snr_data = snr_data.transpose((2, 1, 0))
    new_X = new_X + [snr_data]
    print(eva_iter)
X = np.vstack(new_X)
np.save('../ranker_samples/resnet/resnet_'+str(num_samples)+'.npy', X)
