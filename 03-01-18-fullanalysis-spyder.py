
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
#get_ipython().magic('matplotlib inline')

#%% Single Regression Function

def singleregression(name,dataframe,datlist):
    outfile=open('output/%s_regression.txt' %name,'w')
    for d in datlist:
        outfile.write("\n"+d+"\n")
        
        #outfile.write("\nBetweenness\n")
        form = 'betweenness ~ %s' % (d)
        est = smf.ols(formula=form, data=dataframe).fit()
        d1 = short_summary(est)  

        #outfile.write("\nBonachich\n")
        form = 'BCENT10X ~ %s' % (d)
        est = smf.ols(formula=form, data=dataframe).fit()
        d2 = short_summary(est)  
        
        outfile.write("\nBonachich\n")
        outfile.write(d1.as_text())
        outfile.write("\nBetweenness\n")
        outfile.write(d2.as_text())
            
    outfile.close()   

# ### Computing T Tests for those remainign in Wave4

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

#%%
allWave1.SCID

#%% Longitudinal
#H4IR2
#grouped_home2= homeNominationsMerged2.groupby('SCID')

#H1NB5	Do you usually feel safe in your neighborhood?

slices=[allWave2[['AID','H2IR2']],allWave3[['AID','H3IR2']],allWave4[['AID','H4IR2']]]
val=['H1IR2','H2IR2','H3IR2','H4IR2','SCID','H1NB5']
slice=allWave1[['AID','H1IR2','SCID','H1NB5']]
for data in slices:
    slice=slice.merge(data,on="AID",how="left")
slice = slice[val]
slice = slice.replace(to_replace=6, value=np.nan)
slice = slice.replace(to_replace=8, value=np.nan)
slice = slice.replace(to_replace=9, value=np.nan).dropna()   


#%%

correl=slice.corr()
sns.heatmap(correl, cmap='RdYlGn_r',annot=False)
#
#slice['variance']=slice.var(axis=1)
#slice['mean']=slice.mean(axis=1)
#slice.describe()
correl

#%%Scatter plot matrix

#sns.set(style="ticks")
#sns.pairplot(slice)
#df = sns.load_dataset("iris")
#sns.pairplot(df, hue="species")


#%% GML Variables from Docker Graph-Tool

fname = 'full_gml.gml'

with open(fname) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
lines = [line.rstrip('\n') for line in open(fname)]
data = [x.strip() for x in lines] 
data = [x.replace('"','') for x in data] 

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

df_docker = pd.DataFrame(datadict)
df_docker=df_docker.rename(columns={'label':'AID'})
df_docker=df_docker.astype(float)
df_docker['AID']=df_docker['AID'].astype(int)

# In[23]:

# School extra
# S46B	Since school started this year, how often have you had trouble: paying attention in school?
# S46C	Since school started this year, how often have you had trouble: getting your homework done?
# S46D	Since school started this year, how often have you had trouble: getting along with other students?
# S47	Outside of school hours, about how much time do you spend watching television or video cassettes on an average school day?
# S48	In general, how hard do you try to do your school work well?

schoolMerge=schoolNet.merge(schinfo,on='SCID',how='left')
schoolMergeFull=schoolMerge.merge(inhomeWave4,on="AID",how="outer")
schoolMergeLeft=schoolMerge.merge(inhomeWave4,on="AID",how="left")
schoolMergeLeft=schoolMergeLeft.merge(inschool,on='AID',how="left")
schoolMergeLeft['AID'] = schoolMergeLeft['AID'].astype(int)
schoolMergeLeft=schoolMergeLeft.merge(df_docker,on='AID',how="left")
df_docker
schoolMergeLeft.shape

#%% race

schoolMergeLeft.S6A.astype(int)
#read in CSV
#create dict of dicts - Relation big. Relation Minor. variable:[wave,desc,invalids]


# In[16]:

dna3=pd.read_sas(r"files\DNA\DNA - Wave III\w3dna.xpt",format='xport',encoding='utf-8')
dna4=pd.read_sas(r"files\DNA\DNA - Wave IV\w4dna.xpt",format='xport',encoding='utf-8')

#inhomeWave4
dnamerged=inhomeWave4.merge(dna3,on='AID',how='left')
dnamerged=dnamerged.merge(dna4,on='AID',how='left')
dnamerged=dnamerged.merge(schoolNet,on='AID',how='left')
dnamerged['AID'] = dnamerged['AID'].astype(int)
dnamerged=dnamerged.merge(df_docker,on='AID',how="left")
dnamergedcats=dnamerged.copy()

# In[91]:
dna3list=list(dna3)
dna4list=list(dna4)
dnalist=[]
for d in dna3list:
    if d in dna4list and d != 'AID':
        dnalist.append(d+"_x")
        dnalist.append(d+"_y")
    else:
        dnalist.append(d)

for d in dna4list:
    if d not in dna3list:
        dnalist.append(d)
        
dnalist.remove("AID")

#%%
#dnalist
#%%
#dnamerged[dnalist]=dnamerged[dnalist].astype('category')

for col in dnalist:
    print(col)
    try:
        dnamerged[col]= dnamerged[col].astype('category')
    except:
        print ("Error")

dnamergedcats[dnalist]= dnamerged[dnalist].apply(lambda x: x.cat.codes)

#%%
#singleregression("03-18-DNA-regression",dnamergedcats,dnalist)


#%%
# Slice for intelligence and attractivenesss
#extraW1 = ['AID','H1SE4','H1IR1','H1IR2','H1IR5','H1RM1','H1RF1'] 

extraW1dict={'H1SE4':'intelligent','H1IR1':'physically attractive','H1IR2':'personality attractive',
   'H1IR5':'physical matturity','H1RM1':'mother school','H1RF1':'father school'}
candid = {'H1IR4':'how candid','H1SU8':'how honest self reported'}
   #intelligent, physically attractive,personality attractive,physical matturity
   
extraW4dict={'H4MH7':'intelligent','H4IR1':'physically attractive','H4IR2':'personality attractive',
             'H4MH8':'how attractive self reported'}

education = {'H1RM1':'mother school','H1RF1':'father school'}

race={'S6A':'white','S6B':'black','S6C':'asian','S6D':'indian','S6E':'other'}
          
#w1slice=allWave4[list(extraW4dict.keys())+['AID']]
w1slice=allWave1[list(extraW1dict.keys())+['AID']]
                 
#extra
#6	refused	6	0.04
##8	don't know	32	0.20

#race
#0	not marked	35551	39.45
#1	marked	54567	60.55

#education
#10	He never went to school.	29	0.14
#11	He went to school, but R doesnt know what level.	629	3.03
#96	refused	17	0.08
#97	legitimate skip	6282	30.28
#98	don't know	157	0.76
#99	not applicable	2	0.01

w1slice = w1slice.replace(to_replace=".", value=np.nan)
w1slice = w1slice.replace(to_replace=6, value=0)
w1slice = w1slice.replace(to_replace=8, value=0)
w1slice = w1slice.replace(to_replace=10, value=0)
w1slice = w1slice.replace(to_replace=11, value=0)
w1slice = w1slice.replace(to_replace=96, value=np.nan)
w1slice = w1slice.replace(to_replace=97, value=np.nan)
w1slice = w1slice.replace(to_replace=98, value=np.nan)
w1slice = w1slice.replace(to_replace=99, value=np.nan).dropna()
w1slice['AID']=w1slice['AID'].astype(int)
# need to strip out bad data

schoolMergeLeft=schoolMergeLeft.merge(w1slice,on="AID",how="left")
schoolMergeLeftDroped = schoolMergeLeft[list(extraW1dict.keys())].dropna()
#==============================================================================
# It would be great if we could add a few other variables to the network analysis:
# H1SE4 S9Q4 YOUR INTELLIGENCE-W2
# H1IR1 S39Q1 PHYSICAL ATTRACTIVENESS OF R-W2 - 96,98 - remove
# H1IR2 S39Q2 PERSONALITY ATTRACTNESS OF R-W2
# H1IR5 S39Q5 PHYSICAL MATURITY OF R-W2
# 
# Z-scored average of parental education:
# S12 - H1RM1 S14Q1 RES MOM-EDUCATION LEVEL-  97,98 - remove, 10,11 - to zero
# S18 - H1RF1 S15Q1 RES DAD-EDUCATION LEVEL-W2

#For these other variables, I’d like to see them as (a) main effects, 
#and (b) whether they might act as moderating factors.  
#That is, for example, is the relationship between personality and 
#network structure higher vs. lower depending on whether your parents are more educated?  
#These would take 25 regression analyses, one for each of the 5 variables above 
#for each of the 5 big five personality variables 
#(extraversion, openness, agreeableness, conscientiousness, neuroticism).  
#The regression would have 3 terms: the mean-centered moderator variable, 
#the mean-centered personality variable, and the product of the two mean-centered variables.

#interaction effect
#https://www.theanalysisfactor.com/interpreting-interactions-in-regression/
#fitting with interactions - https://stackoverflow.com/questions/45828964/how-to-add-interaction-term-in-python-sklearn

#==============================================================================

#genderslice
# male/female=S2 1-male 2-female
#schoolMergeLeft=schoolMergeLeft[(schoolMergeLeft['S2']==2)]

#==============================================================================
#%%
schoolMergeLeft['stayed']="Remained"
schoolMergeLeft.loc[schoolMergeLeft['C4VAR001'].isnull(), 'stayed'] = "Left"
schoolMergeLeft['stayed_bin']= 0
schoolMergeLeft.loc[schoolMergeLeft['C4VAR001'].isnull(), 'stayed_bin'] = 1
print('number of participants in wave 1 and 2')
schoolMergeLeft[schoolMergeLeft['stayed']=='Remained'].shape

# ### Wave 4 Personality Variables
# - C4VAR001 COHEN PERCEIVED STRESS SCALE-W4 
# - C4VAR002 CESD DEPRESSION SCALE-W4 
# - C4VAR003 MASTERY SCALE-W4 
# - C4VAR004 EXTRAVERSION-W4 
# - C4VAR005 NEUROTICISM-W4 
# - C4VAR006 AGREEABLENESS-W4 
# - C4VAR007 CONSCIENTIOUSNESS-W4 
# - C4VAR008 OPEN TO EXPER/INTELLECT/IMMAGINATION-W4 
# - C4VAR009 ANXIOUS-W4 
# - C4VAR010 OPTIMISTIC-W4 
# - C4VAR011 ANGRY HOSTILITY-W4 
# 
# ### Wave 1 Network Variables
# - BCENT10X Bonachich centrality
# - REACH maxinum number of steps
# - REACH3 reach in 3 steps

# In[74]:
# ### Computing wave 1 personality variables
def computeWave1(full=False):
    import csv
    
    if full:
        #extract columns from Excel
        w1personalitycolumns=['AID']
        with open('personality_variables.csv', newline='') as csvfile:
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
        
        df_zscore=df_zscore.rename(columns=colsdict)
        df_zscore_raw = pd.concat([df, df_zscore], axis=1)
        df_zscore_raw.head()
        computed_personality=['AID','Sick_RAWavg','Positive_RAWavg','Negative_RAWavg','ProblemSolve_RAWavg','Sick_Zavg','Positive_Zavg','Negative_Zavg','ProblemSolve_Zavg']
        
    
        if not True:
            df_zscore_raw.sort_index(axis=1,inplace=True)
            print(df_zscore_raw.columns.tolist())
            df_zscore_raw.to_csv(r'output\02-03-18-computed-personality.csv')
    
    else:
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df_zscore_raw = pd.DataFrame(data=d)
        computed_personality=[]

    return df_zscore_raw,computed_personality



# In[97]:

#df_zscore_raw,computed_personality = computeWave1()
#schoolMergeLeft.AID=schoolMergeLeft.AID.astype(int)
## schoolMergeLeftComputedZ=schoolMergeLeft.merge(df_zscore_raw,on="AID",how="inner")
#
#print("wave 1 computed")
#print(df_zscore_raw.shape)
#print("Participants in Home and School wave 1")
#print(schoolMergeLeft.shape)

# In[98]:

w4personalitydict={'C4VAR001':'PERCEIVED STRESS','C4VAR002':'DEPRESSION','C4VAR003':'MASTERY SCALE','C4VAR004':'EXTRAVERSION','C4VAR005':'NEUROTICISM','C4VAR006':'AGREEABLENESS','C4VAR007':'CONSCIENTIOUSNESS','C4VAR008':'OPEN TO EXPER/INTELLECT/IMMAGINATION','C4VAR009':'ANXIOUS','C4VAR010':'OPTIMISTIC','C4VAR011':'ANGRY '}
w4personality=['C4VAR001','C4VAR002','C4VAR003','C4VAR004','C4VAR005','C4VAR006','C4VAR007','C4VAR008','C4VAR009','C4VAR010','C4VAR011']
#extraW1
# w4personality=list(w4personalitydict.values())
#s2-gender
#s6a-race

w1network = ['N_ROSTER','IDGX2','ODGX2','BCENT10X','betweenness','S6A','S2','SCID']#,'REACH3']
extra = ['H1SE4','H1IR1'] #intelligent, attractive
correlation = ['SCID','SIZE_x','IDGX2','ODGX2','NOUTNOM','TAB113','BCENT10X','REACH','REACH3','IGDMEAN','PRXPREST','INFLDMN','HAVEBMF','HAVEBFF']
#personalitycorrelation = w1network + w4personality + computed_personality
#computed=w1network+computed_personality
rw1=['Sick_Zavg','Positive_Zavg','Negative_Zavg','ProblemSolve_Zavg']
rw4=['C4VAR004','C4VAR005','C4VAR006','C4VAR007','C4VAR008','C4VAR010']  #extroverted,neurotic,aggreeable,concientious,opentoexperiences,optimistic
#reducedwave1 = w1network + extraW1 + rw1
reducedwave4 = w1network + list(extraW4dict.keys()) + rw4
reducedwave1 = w1network + list(extraW1dict.keys()) + rw4

# In[65]:
schoolMergeRemained=schoolMergeLeft[schoolMergeLeft['stayed']=='Remained']
schoolMergeRemained = schoolMergeRemained[reducedwave1].replace(to_replace=98, value=np.nan).dropna()

#construct gender race variable
schoolMergeRemained['genderrace'] = 'whitemale'
schoolMergeRemained.loc[(schoolMergeRemained['S2']<1.1) & (schoolMergeRemained['S6A']<1), 'genderrace'] = 'blackmale' 
schoolMergeRemained.loc[(schoolMergeRemained['S2']>1.1) & (schoolMergeRemained['S6A']<1), 'genderrace'] = 'blackfemale' 
schoolMergeRemained.loc[(schoolMergeRemained['S2']>1.1) & (schoolMergeRemained['S6A']==1), 'genderrace'] = 'whitefemale' 

schoolMergeRemained=schoolMergeRemained[(schoolMergeRemained['BCENT10X']>.0001)]

#Remove 0 betweenness and log normalize
schoolMergeRemained['betweenness']=schoolMergeRemained['betweenness']*1000000000
schoolMergeRemained=schoolMergeRemained[(schoolMergeRemained['betweenness']>.0001)]#schoolMergeLeft_bcentWave1 = schoolMergeLeft_bcent[extraW1].dropna()
schoolMergeRemained['betweenness']=np.log(schoolMergeRemained['betweenness'])

#Add column for Male/Female
schoolMergeRemained['S2binary']= 0 #female
schoolMergeRemained.loc[(schoolMergeRemained['S2']==1.0), 'S2binary'] = 1 #mail
schoolMergeRemained.S2binary

import statsmodels.formula.api as smf
from IPython.core.display import HTML
def short_summary(est):
    return est.summary().tables[1]

def interactionRegBcent(var1,var2):
    form = 'BCENT10X ~ %s * %s' % (var1,var2)
    est = smf.ols(formula=form, data=schoolMergeRemained).fit()
    return short_summary(est)  
    
def interactionRegBetw(var1,var2):   
    form = 'betweenness ~ %s * %s' % (var1,var2)
    est = smf.ols(formula=form, data=schoolMergeRemained).fit()
    return short_summary(est)  
    

def interactionRegBcentDouble(var1,var2):
    form = 'BCENT10X ~ %s * %s * S6A' % (var1,var2)
    est = smf.ols(formula=form, data=schoolMergeRemained).fit()
    return short_summary(est)  
    
def interactionRegBetwDouble(var1,var2):   
    form = 'betweenness ~ %s * %s * S6A' % (var1,var2)
    est = smf.ols(formula=form, data=schoolMergeRemained).fit()
    return short_summary(est)  
    
#http://blog.datarobot.com/multiple-regression-using-statsmodels


#%%
#schoolMergeLeft_bcent = schoolMergeLeft[['BCENT10X','betweenness']]
#schoolMergeLeft_bcent['betweenlog'] = np.log(schoolMergeLeft_bcent['betweenness'])
#schoolMergeLeft_bcent['betweenness']=schoolMergeLeft_bcent['betweenness']*1000000000
#schoolMergeLeft_bcent=schoolMergeLeft_bcent[(schoolMergeLeft_bcent['BCENT10X']>.0001)]
#schoolMergeLeft_bcent=schoolMergeLeft_bcent[(schoolMergeLeft_bcent['betweenness']>.0001)]
#sns.jointplot(x='BCENT10X',y='betweenlog',data=schoolMergeLeft_bcent,kind='reg') 

#%% exploratory 
sns.lmplot(y='BCENT10X', x="betweenness", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)

#ax.set_titles("Test")  # use this argument literally
#ax.set_axis_labels(x_var="Percentage Depth", y_var="Number of Defects")
#extraW1dict={'H1SE4':'intelligent','H1IR1':'physically attractive','H1IR2':'personality attractive',
#   'H1IR5':'physical matturity','H1RM1':'mother school','H1RF1':'father school'}
#candid = {'H1IR4':'how candid','H1SU8':'how honest self reported'}
#   #intelligent, physically attractive,personality attractive,physical matturity
#   
#extraW4dict={'H4MH7':'intelligent','H4IR1':'physically attractive','H4IR2':'personality attractive',
#             'H4MH8':'how attractive self reported'}

#sns.jointplot(x='H4IR2',y='H4IR1',data=schoolMergeRemained,kind='reg') 
#%%
#create dicts, based on school id
schoolgroups=schoolMergeRemained.groupby("SCID")
for key,values in schoolgroups:
    school_df = schoolgroups.get_group(key)
#    correl=school_df.corr()
#    #sns.heatmap(correl, cmap='RdYlGn_r',annot=False)
#    print (key)
#    print (correl)
    for r in rw4:
        ax=sns.lmplot(y=r, x="betweenness", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
        ax.set_axis_labels(x_var="Betweenness", y_var=w4personalitydict[r])
        ax = plt.gca()
        ax.set_title("School = " + key)
        
    for r in extraW1dict:
        print(extraW1dict[r])
        ax=sns.lmplot(y=r, x="betweenness", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
        ax.set_axis_labels(x_var="Betweenness", y_var=extraW1dict[r])
        ax = plt.gca()
        ax.set_title("School = " + key)   


##   w1network + list(extraW1dict.keys()) + rw4
#    print("RW4")    
#    sns.pairplot(schoolMergeRemained[w1network+rw4+['genderrace']], hue="genderrace")
#    
#    print("extraW1dict")    
#    sns.pairplot(schoolMergeRemained[w1network+extraW1dict.keys()+['genderrace']], hue="genderrace")
##

#%% Regression over dataframe
#schoolMergeRemained
   
for r in rw4:
    print(w4personalitydict[r])
    sns.lmplot(y=r, x="betweenness", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)

#%%
for r in extraW1dict:
    print(extraW1dict[r])
    sns.lmplot(y=r, x="betweenness", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
    
    
#%%
#est = smf.ols(formula='BCENT10X ~ C4VAR002 * H1SE4', data=schoolMergeRemained).fit()

#print(short_summary(est))

outfile=open('output/05_01_network_personality_question_regression.txt','w')
for r in extraW1dict:
    outfile.write("\n=======Race x"+extraW1dict[r]+"========\n")
    d1=interactionRegBcent(r,'S6A')
    d2=interactionRegBetw(r,'S6A')
    outfile.write("\nBonachich\n")
    outfile.write(d1.as_text())
    outfile.write("\nBetweenness\n")
    outfile.write(d2.as_text())    

for r in extraW1dict:
    outfile.write("\n=======Race x"+extraW1dict[r]+"========\n")
    d1=interactionRegBcentDouble(r,'S6A')
    d2=interactionRegBcentDouble(r,'S6A')
    outfile.write("\nBonachich\n")
    outfile.write(d1.as_text())
    outfile.write("\nBetweenness\n")
    outfile.write(d2.as_text())    
    
   
for r in rw4:
    outfile.write("\n======="+w4personalitydict[r]+"xRace=======\n")
    d1=interactionRegBcent(r,'S6A')
    d2=interactionRegBetw(r,'S6A')
    outfile.write("\nBonachich\n")
    outfile.write(d1.as_text())
    outfile.write("\nBetweenness\n")
    outfile.write(d2.as_text())
        
outfile.close()       

#%%



#>>> fig = plt.figure()
#>>> ax = fig.add_subplot(111, projection='3d')
#>>> ax.scatter(df2['Price'],df2['AdSpends'],df2['Sales'],c='blue', marker='o', alpha=0.5)
#>>> ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='None', alpha=0.01)
#>>> ax.set_xlabel('Price')
#>>> ax.set_ylabel('AdSpends')
#>>> ax.set_zlabel('Sales')
#>>> plt.show()

## libraries
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
# 
## Dataset
#df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })
# 
## plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(df['X'], df['Y'], df['Z'], c='skyblue', s=60)
#ax.view_init(30, 185)
#plt.show()


#%% TTests
from scipy.stats import ttest_ind
from numpy import std, mean, sqrt   

def df(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return dof

def cohen_d(x,y):
#     dof=df(x,y)
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

def r_sq(x,y,t):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    r=sqrt (t**2 / (t**2 + dof))
    return r

#cat1 = schoolMergeLeft[schoolMergeLeft['stayed']=='Remained']
#cat2 = schoolMergeLeft[schoolMergeLeft['stayed']=='Left']

#cat1=slice[val[0]]
#cat2=slice[val[1]]

#for var in w1network: #+['S62B','S62E','S62I','S50','S59A','F35']:
#    #are you happy at this school, hows your health, do you smoke, do you feel accepted
#    print("====",var,"====")
#    x=slice[val[0]] #cat1[var]
#    y=slice[val[1]] #cat2[var]
#    tdata=ttest_ind(x, y)
#    print(tdata)
#    print('Cohen d: ',cohen_d(x,y))
#    print('R statistic: ',r_sq(x,y,float(tdata[0])))


#%%

#Cohen’s D is one of the most common ways to measure effect size. An effect size is how large an effect of something is. For example, medication A has a better effect than medication B.
#Small effect = 0.2
#Medium Effect = 0.5
#Large Effect = 0.8
#
#Effect size	r - an effect size is a quantitative measure of the strength of a phenomenon.[1] Examples of effect sizes are the correlation between two variables, the regression coefficient in a regression, the mean difference, or even the risk with which something happens, such as how many people survive after a heart attack for every one person that does not survive.
#Small	0.10
#Medium	0.30
#Large	0.50


x=slice[val[1]] #cat1[var]
y=slice[val[2]] #cat2[var]
tdata=ttest_ind(x, y)
print(tdata)
print('Cohen d: ',cohen_d(x,y))
print('R statistic: ',r_sq(x,y,float(tdata[0])))

#%%

datadna = dnamerged[['DRD4A_x','DRD4A_y','DRD4B_x','DRD4B_y']]