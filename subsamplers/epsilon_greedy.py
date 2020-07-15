# coding: utf-8

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
from keras import layers
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
from keras.optimizers import adagrad
from keras.layers.noise import AlphaDropout
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
epsilon = 0.1
new_X = []

cldnn = load_model('../models/cldnn_ranker.h5')
cnn = load_model('../models/cnn_ranker.h5')
resnet = load_model('../models/resnet_ranker.h5')


def resnet(X_train, X_test):
  Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), range(X_train.shape[0])))
  Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), range(X_test.shape[0])))

  print('training started')
  in_shp = list(X_train.shape[1:])
  print(X_train.shape, in_shp)
  classes = mods
  # Resnet Architecture
  def residual_stack(x):
    def residual_unit(y, _strides=1):
      shortcut_unit = y
      # 1x1 conv linear
      y = layers.Conv1D(32, kernel_size=5, data_format='channels_first',
                        strides=_strides, padding='same', activation='relu')(y)
      y = layers.BatchNormalization()(y)
      y = layers.Conv1D(32, kernel_size=5, data_format='channels_first',
                        strides=_strides, padding='same', activation='linear')(
        y)
      y = layers.BatchNormalization()(y)
      # add batch normalization
      y = layers.add([shortcut_unit, y])
      return y

    x = layers.Conv1D(32, data_format='channels_first', kernel_size=1,
                      padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = residual_unit(x)
    x = residual_unit(x)
    # maxpool for down sampling
    x = layers.MaxPooling1D(data_format='channels_first')(x)
    return x

  inputs = layers.Input(shape=in_shp)
  x = residual_stack(inputs)
  x = residual_stack(x)
  x = residual_stack(
    x)  # Comment this when the input dimensions are 1/32 or lower
  x = residual_stack(
    x)  # Comment this when the input dimensions are 1/16 or lower
  x = residual_stack(
    x)  # Comment this when the input dimensions are 1/8 or lower
  x = Flatten()(x)
  x = Dense(128, kernel_initializer="he_normal", activation="selu",
            name="dense1")(x)
  x = AlphaDropout(0.1)(x)
  x = Dense(128, kernel_initializer="he_normal", activation="selu",
            name="dense2")(x)
  x = AlphaDropout(0.1)(x)
  x = Dense(len(classes), kernel_initializer="he_normal", activation="softmax",
            name="dense3")(x)
  x_out = Reshape([len(classes)])(x)
  model = models.Model(inputs=[inputs], output=[x_out])
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  model.summary()

  # Set up some params
  nb_epoch = 500  # number of epochs to train on
  batch_size = 1024  # training batch size

  # Train the Model
  filepath = 'models/weights_resnet.wts.h5'
  model = multi_gpu_model(model, gpus=3)
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])
  history = model.fit(X_train,
                      Y_train,
                      batch_size=batch_size,
                      epochs=nb_epoch,
                      verbose=2,
                      validation_split=0.25,
                      callbacks=[
                        keras.callbacks.ModelCheckpoint(filepath,
                                                        monitor='val_loss',
                                                        verbose=0,
                                                        save_best_only=True,
                                                        mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10, verbose=0,
                                                      mode='auto')
                      ])
  # we re-load the best weights once training is finished
  model.load_weights(filepath)
  score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
  return score[1]


def e_greedy(k, epsilon, prev_snr_acc, train_set):
  cldnn_list = []
  cnn_list = []
  resnet_list = []
  holistic_list = []

  train_set_copy = deepcopy(train_set)
  for idx in range(X.shape[2]):
    train_set = deepcopy(train_set_copy)
    train_set = train_set.transpose((2, 1, 0))
    new_train_set = np.append(train_set[:idx], np.zeros(
      (1, train_set.shape[1], train_set.shape[2])), axis=0)
    train_set = np.append(new_train_set, train_set[idx + 1:], axis=0)
    train_set = train_set.transpose((2, 1, 0))

    cldnn_score = cldnn.evaluate(train_set, snr_out, batch_size=60000, verbose=0)
    cldnn_list.append((idx, cldnn_score[1]))
    cnn_score = cnn.evaluate(train_set, snr_out, batch_size=60000, verbose=0)
    cnn_list.append((idx, cnn_score[1]))
    resnet_score = resnet.evaluate(train_set, snr_out, batch_size=60000, verbose=0)
    resnet_list.append((idx, resnet_score[1]))

  cldnn_list.sort(key=lambda x: x[1])
  cldnn_list = cldnn_list[:k]
  cldnn_dict = {k: v for k, v in cldnn_list}

  cnn_list.sort(key=lambda x: x[1])
  cnn_list = cnn_list[:k]
  cnn_dict = {k: v for k, v in cnn_list}

  resnet_list.sort(key=lambda x: x[1])
  resnet_list = resnet_list[:k]
  resnet_dict = {k: v for k, v in resnet_list}

  cldnn_cnn_list = [(k, cldnn_dict[k] + cnn_dict[k]) for k in
                    set(cldnn_dict).intersection(cnn_dict)]
  cldnn_cnn_dict = {k: v for k, v in cldnn_cnn_list}
  cnn_resnet_list = [(k, cnn_dict[k] + resnet_dict[k]) for k in
                     set(cnn_dict).intersection(resnet_dict)]
  cnn_resnet_dict = {k: v for k, v in cnn_resnet_list}
  cldnn_resnet_list = [(k, cldnn_dict[k] + resnet_dict[k]) for k in
                       set(cldnn_dict).intersection(resnet_dict)]
  cldnn_resnet_dict = {k: v for k, v in cldnn_resnet_list}

  cldnn_cnn_resnet_list = [(k, cldnn_cnn_dict[k] + resnet_dict[k]) for k in
                           set(cldnn_cnn_dict).intersection(resnet_dict)]
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
  holistic_list = holistic_list[:k]
  holistic_list.sort(key=lambda x: x[1])
  holistic_list.sort(key=lambda x: x[0])

  snr_idxs = [ele[0] for ele in holistic_list]
  train_set = train_set.transpose((2, 1, 0))
  train_set = train_set[snr_idxs]
  train_set = train_set.transpose((2, 1, 0))

  curr_snr_acc = resnet(train_set, X_valid)
  if k == 0:
    if curr_snr_acc > prev_snr_acc:
      return (snr_idxs, curr_snr_acc)
    else:
      return (None, None)
  else:
    train_set_copy = deepcopy(train_set)
    for i in range(min(k, epsilon * 128)):
      train_set = deepcopy(train_set_copy)
      train_set = train_set.transpose((2, 1, 0))
      new_train_set = np.append(train_set[:snr_idxs[i]], np.zeros(
        (1, train_set.shape[1], train_set.shape[2])), axis=0)
      train_set = np.append(new_train_set, train_set[snr_idxs[i] + 1:], axis=0)
      train_set = train_set.transpose((2, 1, 0))

      final_idxs, final_snr_acc = e_greedy(k-1, epsilon, prev_snr_acc, train_set)

      final_idxs.append(snr_idxs[i])

      if final_idxs is not None and final_snr_acc is not None:
        return (final_idxs, final_snr_acc)

    return (None, None)


for eva_iter in range(X.shape[0]//60000):

    # Find sample indices for each SNR
    snr_data = X[eva_iter*60000:(eva_iter+1)*60000]
    snr_out = Y_snr[eva_iter*60000:(eva_iter+1)*60000]

    final_idxs = e_greedy(num_samples, epsilon, 0, snr_data)

    snr_data = snr_data.transpose((2, 1, 0))
    snr_data = snr_data[final_idxs]
    snr_data = snr_data.transpose((2, 1, 0))

    new_X = new_X + [snr_data]
    print(eva_iter)

X = np.vstack(new_X)
np.save('../ranker_samples/e_greedy/e_greedy'+str(num_samples)+'.npy', X)
