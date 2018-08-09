
# coding: utf-8

# In[1]:

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
get_ipython().magic('matplotlib inline')


# In[2]:

inhomeWave4_all = pd.read_sas(r"files\wave4.xpt",format='xport',encoding='utf-8')
inhomeWave4 = pd.read_sas(r"files\W4 Constructed\w4vars.xpt",format='xport',encoding='utf-8')
schinfo = pd.read_sas(r"files\Schinfo.xpt",format='xport',encoding='utf-8')
schoolNet = pd.read_sas(r"files\network.xpt",format='xport',encoding='utf-8')
allWave1 = pd.read_sas(r"files\allwave1.xpt",format='xport',encoding='utf-8')
inschool =  pd.read_sas(r"files\Inschool.xpt",format='xport',encoding='utf-8')


# In[3]:

fname = 'full_gml.gml'

with open(fname) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
lines = [line.rstrip('\n') for line in open(fname)]
data = [x.strip() for x in lines] 
data = [x.replace('"','') for x in data] 


# In[4]:

datadict={"label":[],"betweenness":[],'closeness':[],'eigenvector':[],'hits':[],'katz':[],'pagerank':[]}
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
        


# In[5]:

df = pd.DataFrame(datadict)
df=df.rename(columns={'label':'AID'})
df=df.astype(float)
df['AID']=df['AID'].astype(int)
df.shape


# In[6]:

df2=pd.read_csv("constraint.csv")
df2col=['AID','constraint']
df = df.merge(df2[df2col],on='AID',how="left")
df2.shape


# In[7]:

df.dropna().shape


# In[8]:

# df=df.set_index('aid')
# df['AID']=df['AID'].astype(int)
# df.head()
gmlcolumns=df.columns.tolist()
gmlcolumns


# In[9]:

# read_graph_tool_data
df.dtypes


# In[10]:

inschoolcolumns=inschool.columns.tolist()


# In[11]:

# schoolMerge['AID']=schoolMerge['AID'].astype(int)
# inhomeWave4['AID']=inhomeWave4['AID'].astype(int)

schoolMerge=schoolNet.merge(schinfo,on='SCID',how='left')
schoolMergeFull=schoolMerge.merge(inhomeWave4,on="AID",how="outer")
schoolMergeLeft=schoolMerge.merge(inhomeWave4,on="AID",how="left")
schoolMergeLeft=schoolMergeLeft.merge(inschool,on='AID',how="left")
schoolMergeLeft['AID']=schoolMergeLeft['AID'].astype(int)
schoolMergeLeft.shape


# In[12]:

schoolMergeLeft=schoolMergeLeft.merge(df,on='AID',how="left")
schoolMergeLeft.shape


# In[13]:

schoolMergeLeft.head()


# In[14]:

# Slice for intelligence and attractivenesss
extra = ['AID','H1SE4','H1IR1'] #intelligent, attractive
w1slice=allWave1[extra]
schoolMergeLeft=schoolMergeLeft.merge(w1slice,on="AID",how="left")


# In[15]:

# fullDF['stayed_binary']=1
# fullDF.loc[fullDF['eigen_between_y'].isnull(), 'stayed_binary'] = 0
print ("Personality Vars Exist")
print(schoolMergeLeft[schoolMergeLeft['C4VAR001'].isnull()].shape)
print ("Inbound is blank")
print(schoolMergeLeft[schoolMergeLeft['IDGX2'].isnull()].shape)


# In[16]:

schoolMergeLeft['stayed']="Remained"
schoolMergeLeft.loc[schoolMergeLeft['C4VAR001'].isnull(), 'stayed'] = "Left"

schoolMergeLeft['stayed_bin']= 0
schoolMergeLeft.loc[schoolMergeLeft['C4VAR001'].isnull(), 'stayed_bin'] = 1

print('number of participants in wave 1 and 2')
schoolMergeLeft[schoolMergeLeft['stayed']=='Remained'].shape


# In[ ]:




# In[18]:

import csv

#extract columns from Excel
w1personalitycolumns=['AID']
with open('personality_variables3.csv', newline='') as csvfile:
    w1personalityfile = csv.reader(csvfile)
    w1personality={'Positive':{},"Negative":{},"ProblemSolve":{},"Sick":{}}
    for row in w1personalityfile:
        text=row[0]
        typer=row[1]
        idr=row[2]
        w1personality[typer][idr]=text
        w1personalitycolumns.append(idr)
        
# w1personalitycolumns=list(set(w1personalitycolumns))
print(w1personalitycolumns)
print(w1personality)
# print(allWave1.head())

# create dataframe to work with
df=allWave1[w1personalitycolumns].astype(int)
print(df.shape)
#filter out any where value is a 6 8 9
df = df.replace(to_replace=6, value=np.nan)
df = df.replace(to_replace=7, value=np.nan)
df = df.replace(to_replace=8, value=np.nan)
df = df.replace(to_replace=9, value=np.nan).dropna()
print(df.shape)
#reverse - H1PF16
df['H1PF16']=6-df['H1PF16']
df.dtypes
# w1personalitycolumns
df2=df[w1personalitycolumns[1:]].copy()

colsdict={}
for c in w1personalitycolumns:
    colsdict[c]=c+"_zscore"

df_zscore = (df2 - df2.mean())/df2.std()

zcols=[]
for ids in w1personality:
    columns=['AID']
    print ('==',ids,'==')
    for id in w1personality[ids]:
        columns.append(id)
        
#     print(filtered.head())
    zcalccols=columns[1:]
    zcols.append('%s_avg' % ids)
    df_zscore['%s_Zavg' % ids] = df_zscore[zcalccols].mean(axis=1)
    df['%s_RAWavg' % ids] = df[zcalccols].mean(axis=1)
    
print(zcols)
computed_personality=['AID','Sick_RAWavg','Positive_RAWavg','Negative_RAWavg','ProblemSolve_RAWavg','Sick_Zavg','Positive_Zavg','Negative_Zavg','ProblemSolve_Zavg']
df_zscore=df_zscore.rename(columns=colsdict)
df_zscore_raw = pd.concat([df, df_zscore], axis=1)
df_zscore_raw.head()


# In[44]:

df_zscore_raw.head()


# In[18]:

# df_zscore_raw.to_csv(r'output\02-18-18-computed-personality.csv')


# In[19]:

gmlcolumns


# In[21]:

w4sourcevariables= {
    'C4VAR001':['H4MH3','H4MH4','H4MH5','H4MH6'],
    'C4VAR002':['H4MH18','H4MH19','H4MH21','H4MH22','H4MH26'],
    'C4VAR003':['H4PE37','H4PE38','H4PE39','H4PE40','H4PE41'],
    'C4VAR004':['H4PE1', 'H4PE9', 'H4PE17', 'H4PE25']
    ,'C4VAR005':['H4PE4', 'H4PE12', 'H4PE20', 'H4PE28']
    ,'C4VAR006':['H4PE2', 'H4PE10', 'H4PE18', 'H4PE26'] 
    ,'C4VAR007':['H4PE3', 'H4PE11', 'H4PE19', 'H4PE27'] 
    ,'C4VAR008':['H4PE5', 'H4PE13', 'H4PE21', 'H4PE29'] 
    ,'C4VAR009':['H4PE6','H4PE14','H4PE22','H4PE30']
    ,'C4VAR010':['H4PE7','H4PE15','H4PE23','H4PE31']
    }

w4personality_keys=[]
for w in w4sourcevariables:
    w4personality_keys+=w4sourcevariables[w]
    
# merge4 = inhomeWave4_all.merge(inhomeWave4, how="left", on="AID")
# inhomeWave4['AID']=inhomeWave4['AID'].astype(int)


# In[22]:

w4personality=['C4VAR001','C4VAR002','C4VAR003','C4VAR004','C4VAR005','C4VAR006','C4VAR007','C4VAR008','C4VAR009','C4VAR010','C4VAR011']
w1network = ['N_ROSTER','IDGX2','ODGX2','BCENT10X']#,'REACH3']
extra = ['H1SE4','H1IR1'] #intelligent, attractive
rw1=['Sick_Zavg','Positive_Zavg','Negative_Zavg','ProblemSolve_Zavg']
rw4=['C4VAR002','C4VAR004','C4VAR005','C4VAR006','C4VAR007','C4VAR008']  #depresssion,extroverted,neurotic,aggreeable,concientious,opentoexperiences
reducedwave1 = w1network + extra + rw1
reducedwave4 = w1network + extra + rw4
personalitycorrelation = w1network + w4personality + computed_personality + gmlcolumns
computed=w1network+computed_personality
full_computed=w1network+gmlcolumns


# In[23]:

fullset1 = w1network+gmlcolumns+w1personalitycolumns+computed_personality
fullset2 = w4personality_keys+w4personality


# In[24]:

inhomeWave4_all['AID']=inhomeWave4_all['AID'].astype(int)


# In[25]:

# Create School Merge Z 

schoolMergeLeft.AID=schoolMergeLeft.AID.astype(int)
schoolMergeLeftZ=schoolMergeLeft.merge(df_zscore_raw,on='AID',how='inner')
schoolMergeLeftZ=schoolMergeLeftZ.merge(inhomeWave4_all,on='AID',how='inner')


# In[26]:

# schoolMergeLeftComputedZ=schoolMergeLeft.merge(df_zscore_raw,on="AID",how="inner")
print("wave 1 computed")
print(df_zscore_raw.shape)
print("Participants in Home and School wave 1")
print(schoolMergeLeft.shape)
print('number of participants in school wave 1 and 4')
print(schoolMergeLeft[schoolMergeLeft['stayed']=='Remained'].shape)
print("Participants in Home, School Wave 1")
print(schoolMergeLeftZ.shape)
print("Participants in Home, School Wave 1 and Home Wave 4")
print(schoolMergeLeftZ[schoolMergeLeftZ['stayed']=='Remained'].shape)


# In[27]:

schoolMergeLeftZ.head()


# In[28]:

#remove #98 values from personality analysis
schoolMergeLeftZ[w4personality]=schoolMergeLeftZ[w4personality].replace(to_replace=98, value=np.nan)
schoolMergeLeftZ = schoolMergeLeftZ.dropna(subset=w4personality)
schoolMergeLeftZ.shape


# In[ ]:




# In[30]:

#Removing 0 Bcents

schoolMergeLeft_bcent=schoolMergeLeftZ[(schoolMergeLeftZ['BCENT10X']>.001)]
# schoolMergeLeft_bcent2=schoolMergeLeftZ2[(schoolMergeLeftZ['BCENT10X']>.001)]


# In[31]:

schoolMergeLeft_bcent[gmlcolumns].head()


# In[32]:

#genderslice
# male/female=S2 1-male 2-female
#schoolMergeLeft=schoolMergeLeft[(schoolMergeLeft['S2']==2)]


# In[36]:

sns.jointplot(x='Positive_Zavg',y='Negative_Zavg',data=schoolMergeLeftZ,kind='reg') 


# In[ ]:




# In[34]:

sns.jointplot(y='Positive_Zavg',x='C4VAR004',data=schoolMergeLeft_bcent,kind='reg') 


# In[ ]:

sns.jointplot(x='C4VAR004',y='BCENT10X',data=schoolMergeLeft_bcent,kind='reg') 


# In[ ]:

sns.jointplot(x='Negative_Zavg',y='REACH3',data=schoolMergeLeft_bcent,kind='reg') 


# In[35]:

schoolMergeLeft_bcent[(schoolMergeLeft_bcent['C4VAR010']>50)][['C4VAR004']]


# In[125]:

# schoolMergeLeft_bcent.head()


# In[ ]:




# In[37]:

data=schoolMergeLeft_bcent[personalitycorrelation]
# data.to_csv('output/merged_graph_tool.csv')


# In[38]:

data=schoolMergeLeft_bcent[fullset1+fullset2]
correl=data.corr()
sns.heatmap(correl, cmap='RdYlGn_r',annot=False)
correl.to_csv("output/02-27-full-correlation-out_all.csv")


# In[29]:

# full_computed


# In[245]:

correl=schoolMergeLeft_bcent[full_computed].corr()
sns.heatmap(correl, cmap='RdYlGn_r',annot=False)
correl


# In[ ]:




