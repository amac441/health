# -*- coding: utf-8 -*-
"""
Created on Sun May  6 07:19:53 2018

@author: amac
"""
#S66	What are the most important facts to know about this school?
#S67	What makes this school different from other schools?
#S68 - During the last twelve months, did anything unusual happen at this school? If so, describe what happened.

import networkx as nx
import pandas as pd
import seaborn as sns
import scipy.stats as scs
import numpy as np
from numpy import nan as nan
import networkx as nx
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import zscore
#get_ipython().magic('matplotlib inline')

#%%
schinfo = pd.read_sas(r"files\Schinfo.xpt",format='xport',encoding='utf-8')
schoolNet = pd.read_sas(r"files\network.xpt",format='xport',encoding='utf-8')
inhomeWave4 = pd.read_sas(r"files\W4 Constructed\w4vars.xpt",format='xport',encoding='utf-8')
inschool =  pd.read_sas(r"files\Inschool.xpt",format='xport',encoding='utf-8')

# In[15]:

allWave1 = pd.read_sas(r"files\allwave1.xpt",format='xport',encoding='utf-8')
allWave2 = pd.read_sas(r"files\w2\wave2.xpt",format='xport',encoding='utf-8')
allWave3 = pd.read_sas(r"files\w3\wave3.xpt",format='xport',encoding='utf-8')
allWave4 = pd.read_sas(r"files\wave4.xpt",format='xport',encoding='utf-8')

#%% Longitudinal
inschool.S68