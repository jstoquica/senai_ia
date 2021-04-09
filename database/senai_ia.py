
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3


# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:11:55 2021

@author: juan
"""


# In[ ]:


import time
import pandas as pd
#import dask.dataframe as dd
#from dask_ml.linear_model import LogisticRegression
import numpy as np
from multiprocessing import Queue, Pool, Process, Value
import psutil
import os


# In[ ]:
file = "/home/juan/Desktop/senai-sc/database/database.csv"

#%%
# lendo o dataset com dask
#start_time = time.time()
#data = dd.read_csv(file,sample=80 * 1024 * 1024) #No. Bytes to use 
#print("--- %s seconds ---" % (time.time() - start_time))
#start_time = time.time()
#data.head()
#data.tail()
#print("--- %s seconds ---" % (time.time() - start_time))

#%%

tp_med =  pd.read_csv(file,usecols=[0,1],header=None)

#%%
total_med = np.shape(tp_med)
total_med = total_med[0] - 1

# In[ ]:

#data = dd.read_csv("database.csv",blocksize= 100 * 1000) #max. 64MB não eficiente
range1 = [i for i in range(0,800000-1)]

mean_1=[]
std_1=[]
range_med = range(1,total_med)
#%%

start_time = time.time()

def f(i):

    tp_view = pd.read_csv(file,nrows=1,skiprows=i,usecols=range1,header=None)
    mean_1 = np.mean(tp_view.iloc[0,:])
    std_1 = np.std(tp_view.iloc[0,:])
    return mean_1, std_1

def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(psutil.LOW_PRIORITY_CLASS)

    
if __name__ == '__main__':

    atrib_med = []

    p = Pool(None, limit_cpu)
    atrib_med = p.map(f, range_med)
    p.close()
    # atrib_med = np.array(atrib_med,dtype="object")
print("--- %s seconds ---" % (time.time() - start_time))

#%% Salvar Dataframe para treinamento

dataFrameMD = {'mean': atrib_med[:][0], 'std': atrib_med[:][1]}

#%% sem parallel processing
# mean_2=[]
# std_2=[]
# start_time = time.time()
# for i in range_med:
#     tp_view = pd.read_csv(file,nrows=1,skiprows=i,usecols=[1,2,3,4,5],header=None)
#     mean_2.append(np.mean(tp_view.iloc[0,:]))
#     std_2.append(np.std(tp_view.iloc[0,:])) 
#     print(i)

# print("--- %s seconds ---" % (time.time() - start_time))
# In[ ]:

# lendo o dataset com pandas e chunksize
# start_time = time.time()
# tp = pd.read_csv(file,nrows=210,chunksize=21,header=0,engine='python')
# df = pd.concat(tp)
# print("--- %s seconds ---" % (time.time() - start_time))

# In[ ]:

#Examine the class label imbalance 'target'
df.head()

neg, pos = np.bincount(df['target'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive target: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

# In[ ]:
# Clean, split and normalize the data

cleaned_df = df.copy()

# You don't want the `signal_id` column.
cleaned_df.pop('signal_id')


# In[ ]:

# In[ ]:


# lendo o dataset
#df = pd.read_csv("database.csv",header=None)

