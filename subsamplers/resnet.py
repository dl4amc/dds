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


# In[2]:
# data pre-processing
Xd = cPickle.load(open("../data/RML2016.10b_dict.dat", 'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
print("length of snr",len(snrs))
print("length of mods",len(mods))
X = [] 
Y_snr = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    Y_snr = Y_snr + [mod]*120000
X = np.vstack(X)
Y_snr = np.vstack(Y_snr)
print("shape of X", np.shape(X))
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_snr = to_onehot(map(lambda x: mods.index(lbl[x][0]), range(X.shape[0])))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
num_samples = 64
new_X = []
orig_model = load_model('../models/resnet_ranker.h5')
for eva_iter in range(X.shape[0]//6000):
    snr_data = X[eva_iter*6000:(eva_iter+1)*6000]
    snr_out = Y_snr[eva_iter*6000:(eva_iter+1)*6000]
    snr_acc_list = []
    snr_data_copy = deepcopy(snr_data)
    for idx in range(X.shape[2]):
        snr_data = deepcopy(snr_data_copy)
        snr_data = snr_data.transpose((2,1,0))
    	new_snr_data = np.append(snr_data[:idx], np.zeros((1, snr_data.shape[1], snr_data.shape[2])), axis=0)
    	snr_data = np.append(new_snr_data, snr_data[idx+1:], axis=0)
        snr_data = snr_data.transpose((2,1,0))
        score = orig_model.evaluate(snr_data, snr_out, batch_size=6000, verbose=0)
        snr_acc_list.append((idx, score[1]))
    snr_acc_list.sort(key=lambda x: x[1])
    snr_acc_list = snr_acc_list[:num_samples]
    snr_acc_list.sort(key=lambda x: x[0]) 
    snr_idxs = [ele[0] for ele in snr_acc_list]
    snr_data = snr_data.transpose((2,1,0))
    snr_data = snr_data[snr_idxs]
    snr_data = snr_data.transpose((2,1,0))
    new_X = new_X + [snr_data]
    print(eva_iter)
X = np.vstack(new_X)
np.save('../ranker_samples/resnet/resnet_'+str(num_samples)+'.npy', X)

