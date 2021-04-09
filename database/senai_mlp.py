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

#%%

PATH = '/home/juan/Desktop/senai-sc/database/dataFrameQP.csv'
#%%
origin_data = pd.read_csv(PATH,index_col=0)
#original_data = genfromtxt(f'{PATH}training_data.txt', delimiter=',', dtype='int')
#%% SET GLOABL VARIABLES
DATASET_SIZE = origin_data.shape[0]-1
TRAINING_SET_SIZE = np.round(DATASET_SIZE *0.7,0).astype('int32')
SPLIT_VAR = np.round(DATASET_SIZE-TRAINING_SET_SIZE+1,0).astype('int32')