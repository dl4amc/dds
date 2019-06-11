# coding: utf-8

# In[1]:
# Import all the things we need ---
#get_ipython().magic(u'matplotlib inline')
import os,random
import matplotlib
import numpy as np
matplotlib.use('Agg')
import tensorflow as tf
from keras import layers
from copy import deepcopy
import keras.models as models
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.regularizers import *
import cPickle, random, sys, keras
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
os.environ["KERAS_BACKEND"] = "tensorflow"
K.tensorflow_backend._get_available_gpus()
from keras.optimizers import adam, adagrad
from keras.layers.noise import AlphaDropout
from keras.models import Sequential, load_model, Model
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.similarity_based import fisher_score
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from skfeature.function.statistical_based.chi_square import feature_ranking
from skfeature.function.statistical_based import chi_square as fisher_score
from skfeature.function.statistical_based.chi_square import chi_square as RFS
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Conv2D, MaxPooling2D

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

# In[3]:
# Partition the dataset into training and testing datasets
np.random.seed(2016)     # Random seed value for the partitioning (Also used for random subsampling)
n_examples = X.shape[0]
n_train = n_examples // 2
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
x_train = X[train_idx]
x_test =  X[test_idx]
y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

# compute RFS scores
x_train = np.append(x_train[:,0,:], x_train[:,1,:], axis=1)
score = RFS(abs(x_train), y_train)
idx = feature_ranking(score)
np.save('features/rfs.npy', idx)
print('Features saved')
#idx = np.load('features/rfs.npy', idx)
x_train = x_train.transpose()
x_train = np.split(x_train, 2)
x_train = np.array(x_train).transpose((2,0,1))

# In[4]:
in_shp = list(x_train.shape[1:])
print(x_train.shape, in_shp, snrs)
classes = mods

# create copies of the data
x_train_copy = x_train
y_train_copy = y_train
x_test_copy = x_test
y_test_copy = y_test

# train and compute accuracy of final model trained on selected features
acc_list = []
for img_rows in range(256, 0, -2):
	# load the copies of the original data
	x_train = x_train_copy
	y_train = y_train_copy
	x_test = x_test_copy
	y_test = y_test_copy
	
	# load the selected features
	x_train = np.append(x_train[:,0,:], x_train[:,1,:], axis=1) 
	x_train = x_train[:, idx[0:img_rows]]
	x_train = x_train.transpose()
	x_train = np.split(x_train, 2)
	x_train = np.array(x_train).transpose((2,0,1))
	
	x_test = np.append(x_test[:,0,:], x_test[:,1,:], axis=1) 
	x_test = x_test[:, idx[0:img_rows]]
	x_test = x_test.transpose()
	x_test = np.split(x_test, 2)
	x_test = np.array(x_test).transpose((2,0,1))

	# final model Resnet Architecture
	def residual_stack(x):
		def residual_unit(y,_strides=1):
    			shortcut_unit=y
    			# 1x1 conv linear
    			y = layers.Conv1D(32, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='relu')(y)
    			y = layers.BatchNormalization()(y)
    			y = layers.Conv1D(32, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='linear')(y)
    			y = layers.BatchNormalization()(y)
    			# add batch normalization
    			y = layers.add([shortcut_unit,y])
    			return y
  
		x = layers.Conv1D(32, data_format='channels_first',kernel_size=1, padding='same',activation='linear')(x)
		x = layers.BatchNormalization()(x)
		x = residual_unit(x)
		x = residual_unit(x)
		# maxpool for down sampling
		x = layers.MaxPooling1D(data_format='channels_first')(x)
		return x

	inputs=layers.Input(shape=list(x_train.shape[1:]))
	x = residual_stack(inputs)
	x = residual_stack(x)
	#x = residual_stack(x)    # Comment this when the input dimensions are 1/32 or lower
	#x = residual_stack(x)    # Comment this when the input dimensions are 1/16 or lower
	#x = residual_stack(x)    # Comment this when the input dimensions are 1/8 or lower
	x = Flatten()(x)
	x = Dense(128,kernel_initializer="he_normal", activation="selu", name="dense1")(x)	
	x = AlphaDropout(0.1)(x)
	x = Dense(128,kernel_initializer="he_normal", activation="selu", name="dense2")(x)
	x = AlphaDropout(0.1)(x)
	x = Dense(len(classes),kernel_initializer="he_normal", activation="softmax", name="dense3")(x)
	x_out = Reshape([len(classes)])(x)
	model = models.Model(inputs=[inputs], output=[x_out])
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.summary()
	# Set up some params 
	nb_epoch = 500     # number of epochs to train on
	batch_size = 1024  # training batch size

	# In[7]:
	# Train the Model
	# perform training ...
	#   - call the main training loop in keras for our network+dataset
	filepath = 'models/weights_resnet.wts.h5'
	model = multi_gpu_model(model, gpus=3)
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_split=0.2,
        callbacks = [
        	keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        	keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        ])
	# we re-load the best weights once training is finished
	model.load_weights(filepath)

	# In[8]:
	# Evaluate and Plot Model Performance
	# Show simple version of performance
	score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
	print(score)

	# In[12]:
	# Print acciracies for each snr
	acc = {}
	for snr in snrs:
		# extract classes @ SNR
		test_SNRs = map(lambda x: lbl[x][1], test_idx)
		test_X_i = x_test[np.where(np.array(test_SNRs)==snr)]
		test_Y_i = y_test[np.where(np.array(test_SNRs)==snr)]    

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
	#plt.figure()
	#plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
	cor = np.sum(np.diag(conf))
	ncor = np.sum(conf) - cor
	print cor*100 / (cor+ncor)
	acc[snr] = 1.0*cor/(cor+ncor)
	if snr == 18:
		acc_list.append(cor*100 / (cor+ncor))

# print final model accuracies for each feature count
for acc_value in acc_list:
	print(acc_value)

