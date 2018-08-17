
# coding: utf-8

# ## There are 3 different network files
# - In school interview wave 1 (inschoolNominations1)
# - In home interview wave 1
# - In home interview wave 2
# 
# ### In School Wave 1 are used to construct the network variables
# - network.xpt
# 
# ### We want to try and recreate the network variables for wave 1 in school network, and then apply those mappings to in home networks
# 
# ### Note that inHome and inSchool use different variables
#  - inSchool - school id  - SSCHLCDE
#  - network.xpt - student id - SQID (converts to Aid using inSchool)
#  - inHome - school id - SCID

import os
os.getcwd()

# In[3]:

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
import statsmodels.formula.api as smf
from IPython.core.display import HTML
#get_ipython().magic('matplotlib inline')

#%%
import datetime


#assert diff_month(datetime(2010,10,1), datetime(2010,9,1)) == 1
#assert diff_month(datetime(2010,10,1), datetime(2009,10,1)) == 12
#assert diff_month(datetime(2010,10,1), datetime(2009,11,1)) == 11
#assert diff_month(datetime(2010,10,1), datetime(2009,8,1)) == 14

#%%
def filterout(name,dataframe,columnname,filterdict):
    sizes={}
    sizes['original']=len(dataframe.index)
    
    for key in filterdict:
        dataframe[columnname] = dataframe[columnname].replace(to_replace=filterdict[key], value=np.nan)#dataframe[columnname].map({filterdict[key]: np.nan})
        dataframe=dataframe.dropna()
        sizes[key]=len(dataframe.index)
    
    print(name)
    print(sizes)
    return dataframe

#%%
schinfo = pd.read_sas(r"files\Schinfo.xpt",format='xport',encoding='utf-8')
#%%
#schoolsize and such
skinfo=['SCID','SIZE','METRO','REGION','GRADES','SCHTYPE']
schinfo = schinfo[skinfo].dropna()
schinfo = filterout("schoolsize",schinfo,'SIZE',{'duplicate':'!'})
#%%
skinfo.pop(0)
for d in skinfo:
    schinfo[d]=schinfo[d].astype(int)

#SCHTYPE
#1 public
#2 Catholic
#3 private

#High School Stratification Size SIZE num 1
#4 1 125 or fewer students
#13 2 126-350 students
#35 3 351-775 students
#80 4 776 or more students
#40 ! duplicate schools in strata, not part of main study
#
#High School Stratification Metropolitan Location METRO num 1
#40 1 urban
#73 2 suburban
#19 3 rural

#%%
w4personalitydict={'C4VAR001':'perceived_stress','C4VAR002':'depression','C4VAR003':'mastery','C4VAR004':'extraversion',
                   'C4VAR005':'neuroticism','C4VAR006':'agreeableness','C4VAR007':'concientiousness',
                   'C4VAR008':'oppenness','C4VAR009':'anxiousness','C4VAR010':'optimism','C4VAR011':'anger '}
data=[*w4personalitydict]
#%% PERSONALITY CONSTRUCTS (WAVE 4)

inhomeWave4 = pd.read_sas(r"files\W4 Constructed\w4vars.xpt",format='xport',encoding='utf-8')[data+['AID']]
for d in data:
    inhomeWave4 = filterout(w4personalitydict[d],inhomeWave4,d,{'unknown':98})
         
#%% NETWORK
schoolNet = pd.read_sas(r"files\network.xpt",format='xport',encoding='utf-8')[['AID', 'SCID', 'SIZE', 'IDGX2', 'ODGX2', 'NOUTNOM', 'TAB113',
       'BCENT10X', 'REACH', 'REACH3']]

#%% IN SCHOOL

inschool =  pd.read_sas(r"files\Inschool.xpt",format='xport',encoding='utf-8')[['AID','S2','S6A','S6B','S6C','S6D','S6E']].dropna()
#S2 -  1-male 2-female
#S6a - race (are you white - 1 is yes)
inschool=filterout('MultGender',inschool,'S2',{'gender':9})
inschool['S6A']=inschool.S6A.astype(int)
inschool['S6B']=inschool.S6B.astype(int)
inschool['S6C']=inschool.S6C.astype(int)
inschool['S6D']=inschool.S6D.astype(int)
inschool['S6E']=inschool.S6E.astype(int)

#Merge Native and Other
inschool['S6E']=inschool.S6E + inschool.S6D
inschool.loc[(inschool['S6E']==2),'S6E']=1
#schoolMergeRemained.loc[(schoolMergeRemained['S2']==1.0), 'S2binary'] = 1 #mail

#%% WAVE1
#attractive=['H%sIR1':'physically attractive','H%sIR2':'personality attractive']
allWave1 = pd.read_sas(r"files\w1\wave1.xpt",format='xport',encoding='utf-8')[['AID','H1IR2','H1GI1Y','H1GI1M','IMONTH','IYEAR']].dropna()

#%%  CALCULATE AGE

allWave1time=allWave1.copy()
allWave1.dtypes
allWave1time['IYEAR']=allWave1time['IYEAR'].astype(int)+1900
allWave1time['IMONTH']=allWave1time['IMONTH'].astype(int)
allWave1time['H1GI1Y']=allWave1time['H1GI1Y'].astype(int)+1900
allWave1time['H1GI1M']=allWave1time['H1GI1M'].astype(int)
allWave1time['DateOfInt'] = allWave1time['IMONTH'].astype(str) + '/25/' + allWave1time['IYEAR'].astype(str) 
allWave1time['DOB'] = allWave1time['H1GI1M'].astype(str) + '/25/' + allWave1time['H1GI1Y'].astype(str) 
allWave1time['DateOfInt']=pd.to_datetime(allWave1time.DateOfInt,format="%m/%d/%Y")
allWave1time['DOB']=pd.to_datetime(allWave1time.DOB,format="%m/%d/%Y",errors="coerce")
allWave1time=allWave1time.dropna()
allWave1time.dtypes
#allWave1time['age'] = pd.Timedelta(allWave1time.DateOfInt - allWave1time.DOB).days / 365.25
allWave1time['age'] = (allWave1time.DateOfInt - allWave1time.DOB).astype('timedelta64[h]')/(24*365.25)
allWave1time['ageint']=allWave1time['age'].astype(int)


#%%
#H1IR2 / H4IR2 - how attractive is the respondants personality (filter 6,8,9)
personfilter={'refused':6,'dontknow':8,'not applicable':9}
agefilter={'young':74,'young2':75,'old2':83,'old3':96}
allWave1_2=filterout('wave1-attractiveness',allWave1time,'H1IR2',personfilter)
allWave1_2=filterout('wave1-age',allWave1_2,'H1GI1Y',agefilter)

#%%

allWave4 = pd.read_sas(r"files\w4\wave4.xpt",format='xport',encoding='utf-8')[['AID','H4IR2']]
allWave4_2=filterout('wave4-attractiveness',allWave4,'H4IR2',personfilter)


#%% GML Variables from Docker Graph-Tool

fname = 'full_gml.gml'

with open(fname) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
lines = [line.rstrip('\n') for line in open(fname)]
data = [x.strip() for x in lines] 
data = [x.replace('"','') for x in data] 

datadict={"label":[],"betweenness":[],'closeness':[],'eigenvector':[],'hits':[],'katz':[],'pagerank':[],'id':[]}
capture=False
for d in data:
    
    if d==']':
        capture=False
    
    if capture:
        try:
            values = d.split(' ')
            datadict[values[0]].append(values[1])
        except:
            print ("pass",d)
    
    if "node" in d:
        capture=True

df_docker = pd.DataFrame(datadict)
df_docker=df_docker.rename(columns={'label':'AID'})
df_docker=df_docker.astype(float)
df_docker['AID']=df_docker['AID'].astype(int)


#=================================
#%%  MERGING EVERYTHING TOGETHER #
#=================================

schoolMerge=schoolNet.merge(schinfo,on='SCID',how='inner')
#inhomewave4 has personality metrics
#schoolMergeFull=schoolMerge.merge(inhomeWave4,on="AID",how="outer")
schoolMergeLeft=schoolMerge.merge(inhomeWave4,on="AID",how="inner")
#network metrics
schoolMergeLeft=schoolMergeLeft.merge(inschool,on='AID',how="inner")
schoolMergeLeft=schoolMergeLeft.merge(allWave4_2,on="AID",how="inner")
schoolMergeLeft=schoolMergeLeft.merge(allWave1_2,on="AID",how="inner")
schoolMergeLeft['AID'] = schoolMergeLeft['AID'].astype(int)
schoolMergeDocker=schoolMergeLeft.merge(df_docker,on='AID',how="inner")
schoolMergeLeft.shape

#%%
sml=schoolMergeDocker[['IDGX2','ODGX2','BCENT10X','betweenness']]
sns.jointplot(x='BCENT10X',y='betweenness',data=sml,kind='reg') 

#%%
#sns.countplot(x="betweenness", data=sml)

#%% FILTERING

#filter out bad personality
#filter out bad race or gender

#filter Betweenness
schoolMergeRemained=schoolMergeLeft[(schoolMergeLeft['BCENT10X']>.0001)]
schoolMergeRemained['log_bonachich']=np.log(schoolMergeRemained['BCENT10X'])

schoolMergeRemained['IDGX2']=schoolMergeRemained['IDGX2'].astype(int)

#create merged race
schoolMergeRemained['genderrace'] = 'whitemale'
schoolMergeRemained.loc[(schoolMergeRemained['S2']<1.1) & (schoolMergeRemained['S6A']<1), 'genderrace'] = 'blackmale' 
schoolMergeRemained.loc[(schoolMergeRemained['S2']>1.1) & (schoolMergeRemained['S6A']<1), 'genderrace'] = 'blackfemale' 
schoolMergeRemained.loc[(schoolMergeRemained['S2']>1.1) & (schoolMergeRemained['S6A']==1), 'genderrace'] = 'whitefemale' 

schoolMergeRemained.shape
#%% RENAMING
keylist=list(v1.keys())+['IDGX2','ODGX2']

#%%

v1={'S6A':'white','S6B':'black','S6C':'asian','S6D':'native','S6E':'other','S2':'gender','H4IR2':'personality_attractive','BCENT10X':'bonachich'}
v1.update(w4personalitydict)
schoolMergeRemained[keylist]=schoolMergeRemained[keylist].apply(zscore)
schoolMergeZ=schoolMergeRemained.rename(index=str,columns=v1)

#%%
#=================================
#
#Exporting SchoolMergeZ
#After this can be a whole new file
#
#=================================

schoolMergeZ.to_csv("files/mergedData/MERGED_extroversion_agreeableness_full.csv")
