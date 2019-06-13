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

model_path = 'models/final_model.h5'

# In[2]:
# Dataset setup
Xd = cPickle.load(open("data/RML2016.10b_dict.dat",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
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
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_snr = to_onehot(map(lambda x: mods.index(lbl[x][0]), range(X.shape[0])))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''num_samples = 64
snr_divs = 6000
new_X = []
orig_model = load_model('models/model_cldnn_original.h5')
for eva_iter in range(X.shape[0]//snr_divs):
    snr_data = X[eva_iter*snr_divs:(eva_iter+1)*snr_divs]
    snr_out = Y_snr[eva_iter*snr_divs:(eva_iter+1)*snr_divs]
    snr_acc_list = []
    snr_data_copy = deepcopy(snr_data)
    for idx in range(X.shape[2]):
        snr_data = deepcopy(snr_data_copy)
        snr_data = snr_data.transpose((2,1,0))
    	new_snr_data = np.append(snr_data[:idx], np.zeros((1, snr_data.shape[1], snr_data.shape[2])), axis=0)
    	snr_data = np.append(new_snr_data, snr_data[idx+1:], axis=0)
        snr_data = snr_data.transpose((2,1,0))
        score = orig_model.evaluate(snr_data, snr_out, batch_size=snr_divs, verbose=0)
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
np.save('dds_samples/dds_samples_1_2.npy', X)'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#X = np.load('dds_samples/dds_samples_1_2.npy')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In[3]:
# Partition the dataset into training and testing datasets
np.random.seed(2016)     # Random seed value for the partitioning (Also used for random subsampling)
n_examples = X.shape[0]
n_train = n_examples // 2
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

# In[4]:
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp, snrs)
classes = mods

# In[5]:
dr = 0.6 # dropout rate (%)
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(kernel_initializer="glorot_uniform", name="conv1", activation="relu", data_format="channels_first", padding="valid", filters=256, kernel_size=(1, 3)))
model.add(Dropout(dr))

model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(kernel_initializer="glorot_uniform", name="conv2", activation="relu", data_format="channels_first", padding="valid", filters=256, kernel_size=(2, 3)))
model.add(Dropout(dr))

model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(kernel_initializer="glorot_uniform", name="conv3", activation="relu", data_format="channels_first", padding="valid", filters=80, kernel_size=(1, 3)))
model.add(Dropout(dr))

model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(kernel_initializer="glorot_uniform", name="conv4", activation="relu", data_format="channels_first", padding="valid", filters=80, kernel_size=(1, 3)))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))

model.add(Flatten())
# 128 (1) -> 11200; 64 (1/2) -> 6080; 32 (1/4) -> 3520; 16 (1/8) -> 2240; 8 (1/16) -> 1600; 4 (1/32) -> 1280
model.add(Reshape((1,2240)))
model.add(keras.layers.LSTM(50))
model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), init='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
#opt=adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# In[6]:
# Set up some params
nb_epoch = 500     # number of epochs to train on
batch_size = 1024  # training batch size

# In[7]:
# Train the Model
filepath = 'models/weights_cldnn.wts.h5'
model = multi_gpu_model(model, gpus=3)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)

# Save the model (architecture and weights)
model.save(model_path)

# In[8]:
# Evaluate and Plot Model Performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(score)

# In[12]:
# Plot confusion matrix
acc = {}
for snr in snrs:
    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print cor*100 / (cor+ncor)
    acc[snr] = 1.0*cor/(cor+ncor)

