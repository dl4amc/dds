# coding: utf-8

# In[1]:
# Import all the things we need ---
#get_ipython().magic(u'matplotlib inline')
import os,random
#os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)   #disabled because we do not have a hardware GPU
import numpy as np
from copy import deepcopy
#import theano as th
#import theano.tensor as T
from keras.utils import np_utils
from keras.models import load_model
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
from keras.optimizers import adagrad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
import cPickle, random, sys, keras
from keras.utils import multi_gpu_model
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import tensorflow as tf


# Dataset setup
Xd = cPickle.load(open("../data/RML2016.10b_dict.dat", 'rb'))
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
Y_snr = []
lbl = []
for snr in snrs:
    for mod in mods:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
        Y_snr = Y_snr + [mod]*6000
X = np.vstack(X)
Y_snr = np.vstack(Y_snr)


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


Y_snr = to_onehot(map(lambda x: mods.index(lbl[x][0]), range(X.shape[0])))

# Use only the train split
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples // 2
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
X = X[train_idx]

print("shape of X", np.shape(X))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
num_samples = 64
new_X = []

cldnn = load_model('../models/cldnn_ranker.h5')
cnn = load_model('../models/cnn_ranker.h5')
resnet = load_model('../models/resnet_ranker.h5')

for eva_iter in range(X.shape[0]//60000):
    snr_data = X[eva_iter*60000:(eva_iter+1)*60000]
    snr_out = Y_snr[eva_iter*60000:(eva_iter+1)*60000]

    cldnn_list = []
    cnn_list = []
    resnet_list = []
    holistic_list = []

    snr_data_copy = deepcopy(snr_data)
    for idx in range(X.shape[2]):
        snr_data = deepcopy(snr_data_copy)
        snr_data = snr_data.transpose((2, 1, 0))
        new_snr_data = np.append(snr_data[:idx], np.zeros((1, snr_data.shape[1], snr_data.shape[2])), axis=0)
        snr_data = np.append(new_snr_data, snr_data[idx+1:], axis=0)
        snr_data = snr_data.transpose((2, 1, 0))

        cldnn_score = cldnn.evaluate(snr_data, snr_out, batch_size=60000, verbose=0)
        cldnn_list.append((idx, cldnn_score[1]))
        cnn_score = cnn.evaluate(snr_data, snr_out, batch_size=60000, verbose=0)
        cnn_list.append((idx, cnn_score[1]))
        resnet_score = resnet.evaluate(snr_data, snr_out, batch_size=60000, verbose=0)
        resnet_list.append((idx, resnet_score[1]))

    cldnn_list.sort(key=lambda x: x[1])
    cldnn_list = cldnn_list[:num_samples]
    cldnn_dict = {k: v for k, v in cldnn_list}

    cnn_list.sort(key=lambda x: x[1])
    cnn_list = cnn_list[:num_samples]
    cnn_dict = {k: v for k, v in cnn_list}

    resnet_list.sort(key=lambda x: x[1])
    resnet_list = resnet_list[:num_samples]
    resnet_dict = {k: v for k, v in resnet_list}

    cldnn_cnn_list = [(k, cldnn_dict[k] + cnn_dict[k]) for k in set(cldnn_dict).intersection(cnn_dict)]
    cldnn_cnn_dict = {k: v for k, v in cldnn_cnn_list}
    cnn_resnet_list = [(k, cnn_dict[k] + resnet_dict[k]) for k in set(cnn_dict).intersection(resnet_dict)]
    cnn_resnet_dict = {k: v for k, v in cnn_resnet_list}
    cldnn_resnet_list = [(k, cldnn_dict[k] + resnet_dict[k]) for k in set(cldnn_dict).intersection(resnet_dict)]
    cldnn_resnet_dict = {k: v for k, v in cldnn_resnet_list}

    cldnn_cnn_resnet_list = [(k, cldnn_cnn_dict[k] + resnet_dict[k]) for k in set(cldnn_cnn_dict).intersection(resnet_dict)]
    cldnn_cnn_resnet_list.sort(key=lambda x: x[1])

    tier_1_samples = cldnn_cnn_resnet_list
    tier_1 = [ele[0] for ele in tier_1_samples]

    for ele in tier_1:
        del cldnn_cnn_dict[ele]
        del cnn_resnet_dict[ele]
        del cldnn_resnet_dict[ele]

    cldnn_cnn_list = [(k, v) for k, v in cldnn_cnn_dict]
    cnn_resnet_list = [(k, v) for k, v in cnn_resnet_dict]
    cldnn_resnet_list = [(k, v) for k, v in cldnn_resnet_dict]

    tier_2_samples = cldnn_cnn_list + cnn_resnet_list + cldnn_resnet_list
    tier_2_samples.sort(key=lambda x: x[1])

    tier_2 = [ele[0] for ele in tier_2_samples]

    for ele in tier_1:
        del cldnn_dict[ele]
        del cnn_dict[ele]
        del resnet_dict[ele]

    for ele in tier_2:
        del cldnn_dict[ele]
        del cnn_dict[ele]
        del resnet_dict[ele]

    cldnn_list = [(k, v) for k, v in cldnn_dict]
    cnn_list = [(k, v) for k, v in cnn_dict]
    resnet_list = [(k, v) for k, v in resnet_dict]

    tier_3_samples = cldnn_list + cnn_list + resnet_list
    tier_3_samples.sort(key=lambda x: x[1])

    holistic_list = holistic_list + [tier_1_samples]
    holistic_list = holistic_list + [tier_2_samples]
    holistic_list = holistic_list + [tier_3_samples]
    holistic_list = holistic_list[:num_samples]
    holistic_list.sort(key=lambda x: x[1])
    holistic_list.sort(key=lambda x: x[0])

    snr_idxs = [ele[0] for ele in holistic_list]
    snr_data = snr_data.transpose((2, 1, 0))
    snr_data = snr_data[snr_idxs]
    snr_data = snr_data.transpose((2, 1, 0))
    new_X = new_X + [snr_data]
    print(eva_iter)

X = np.vstack(new_X)
np.save('../ranker_samples/holistic/holistic_'+str(num_samples)+'.npy', X)
