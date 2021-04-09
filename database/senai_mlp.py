#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:28:04 2021

@author: juan
"""

#%%
import numpy as np
from numpy import genfromtxt
import pandas as pd
#import matplotlib.pyplot as plt
#%%
from keras.optimizers import Adam
from keras.models import  Model, Sequential
from keras.layers import Dense, LSTM, Activation, GRU, Dropout,Conv1D, BatchNormalization
from keras.layers import MaxPooling1D, Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Nadam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adamax
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import ReLU
from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import ThresholdedReLU
import keras as kr

#%%

PATH = '/home/juan/Desktop/senai-sc/database/dataFrameQP.csv'
#%%
origin_data = pd.read_csv(PATH,index_col=0)
#original_data = genfromtxt(f'{PATH}training_data.txt', delimiter=',', dtype='int')
#%% SET GLOABL VARIABLES
DATASET_SIZE = origin_data.shape[0]-1
TRAINING_SET_SIZE = np.round(DATASET_SIZE *0.7,0).astype('int32')
SPLIT_VAR = np.round(DATASET_SIZE-TRAINING_SET_SIZE+1,0).astype('int32')

#%%

#%%floa
training_set = []
for i in range(len(origin_data) - SPLIT_VAR):
    training_set.append(origin_data.iloc[i,:])
training_set = np.array(training_set)

#%%
train_input = []
train_output = []
for i in range(0,TRAINING_SET_SIZE):
    train_input.append(training_set[i,0:3])
    train_output.append(training_set[i,3:4])
    
#%%

train_input = np.array(train_input)
train_output = np.array(train_output)

train_input = train_input.reshape((train_input.shape[0], train_input.shape[1], 1))
train_output = train_output.reshape((train_output.shape[0], train_output.shape[1], 1))

#%%
train_input = train_input.transpose()
train_output = train_output.transpose()

#%%

#%%
test_input = []
test_output = []
for i in range(TRAINING_SET_SIZE, len(origin_data)):
    test_input.append(origin_data.iloc[i,0:3])
    test_output.append(origin_data.iloc[i,3:4])
    
#%%
test_input = np.array(test_input)
test_output = np.array(test_output)

test_input = test_input.reshape((test_input.shape[0], test_input.shape[1], 1))
test_output = test_output.reshape((test_output.shape[0], test_output.shape[1], 1))

#%%

test_input = test_input.transpose()
test_output = test_output.transpose()

#%%
#%%
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.5,
                                         patience=3,
                                         cooldown=2,
                                         min_lr=0.001,
                                         verbose=1)

callbacks = [reduce_learning_rate]
#%%

sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False) #33%
rmsProp = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) #99.2
adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0) #99.5
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0) #99.60
adam = Adam(lr=0.01, beta_1=0.9,beta_2=0.999,epsilon=1e-12,decay=0.0) #99.6
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) #99.6
nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004) #99.6
#%%

leaky = LeakyReLU(alpha=0.4) #99.84 %
#prelu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
relu = ReLU(max_value=None, negative_slope=0.0, threshold=0.0) #66 %
elu = ELU(alpha=1.0)
threlu = ThresholdedReLU(theta=1.0) #66%

#%%
s = (train_input.shape[1],train_input.shape[2])

#%%
#%%
model = Sequential() # The Keras Sequential model is a linear stack of layers.
model.add(Dense(3,kernel_initializer='normal',input_shape=s )) # Dense layer
layer = Dense(150)
layer.activation = leaky
model.add(layer)
model.add(Dense(3, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer=adamax, metrics=['accuracy']) # Using logloss
#    return model

#%%

#%%
model.summary()
print("Inputs: {}".format(model.input_shape))
print("Outputs: {}".format(model.output_shape))
print("Actual input: {}".format(train_input.shape))
print("Actual output: {}".format(train_output.shape))
kr.utils.plot_model(model,'/home/juan/Documents/GitHub/senai_ia/database/topol_MLP.jpeg')

#%%































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    