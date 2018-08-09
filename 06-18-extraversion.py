
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
#schoolMergeLeft=schoolMergeLeft.merge(df_docker,on='AID',how="left")
schoolMergeLeft.shape

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

#%% GETTING HISTOS
sns.set(style="darkgrid")
#sns.countplot(x="gender", hue="race", data=schoolMergeRemained)
#%%
sns.countplot(x="SIZE_y", hue="METRO", data=schoolMergeZ)
#%%
sns.countplot(x="GRADES", data=schoolMergeZ)
#%%
sns.countplot(x="ageint", data=schoolMergeZ)
#%%

#Grade span of school GRADES num 2
#13 1 grades K-12
#14 2 grades 7-8
#73 3 grades 9-12
#1 4 grades 7-9
#6 5 grades 10-12
#9 6 grades 7-12
#5 7 grades special
#35 8 grades 6-8
#2 9 grades 8-12
#7 10 grades K-8
#1 11 grades 6-13+
#1 12 grades 7-13+
#2 13 grades 5-8
#1 14 grades 6-9
#1 15 grades K-13+
#1 16 grades 5-7

#%%

#==============
#Plots
#==============

ax=sns.lmplot(x='agreeableness', y="log_bonachich", hue="genderrace",data=schoolMergeZ,fit_reg=True, truncate=True)
ax.set_axis_labels(x_var="age",y_var="bonachich")
ax = plt.gca()
ax.set_title("Agreeableness and Bonachich")

#%%
ax=sns.lmplot(x='personality_attractive', y="bonachich", hue="genderrace",data=schoolMergeZ,fit_reg=True, truncate=True)
ax.set_axis_labels(x_var="personality_attractive",y_var="bonachich")
ax = plt.gca()
ax.set_title("personality_attractive and Bonachich")

#%%
sns.jointplot(x='extraversion',y='log_bonachich',data=schoolMergeZ,kind='reg') 
ax.set_title("Extraversion and Log Bonachich")

#%%
sns.jointplot(x='extraversion',y='bonachich',data=schoolMergeZ,kind='reg') 
ax.set_title("personality_attractive and Bonachich")
ax.set_title("Extraversion and Bonachich")

#%%
race=['white','black','asian','other']
xvar='agreeableness'
yvarlist=['bonachich','IDGX2']
huelist=['SIZE_y','REGION','METRO','genderrace','gender','ageint']
for yvar in yvarlist:
    print (yvar)
    for huevar in huelist:
        print (huevar)
        ax=sns.lmplot(x=xvar, y=yvar, hue=huevar,data=schoolMergeZ,fit_reg=True, truncate=True)
        ax = plt.gca()
        fig = ax.get_figure()
        name=yvar+"_"+huevar
        fig.savefig("output/%s.png" % name)
#
#ax=sns.lmplot(y='H4IR1', x="H4IR2", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
#ax.set_axis_labels(y_var="physically attractive", x_var='personality attractive')
#ax = plt.gca()
#ax.set_title("Personality vs Physhical Attractive")
#fig = ax.get_figure()
#fig.savefig("schoolplots/PersonalityVPhysical.png")


#%%
sns.jointplot(x='METRO',y='SIZE_y',data=schoolMergeZ,kind='reg') 

#%%
#High School Stratification Region REGION num 1
#28 1 West
#30 2 Midwest
#54 3 South
#20 4 Northeast


#High School Stratification Metropolitan Location METRO num 1
#40 1 urban
#73 2 suburban
#19 3 rural

#Large vs small school
#4 1 125 or fewer students
#13 2 126-350 students
#35 3 351-775 students
#80 4 776 or more students
#40 ! duplicate schools in strata, not part of main study
zlarge=schoolMergeZ[(schoolMergeZ['SIZE_y']>2.1)]
zsmall=schoolMergeZ[(schoolMergeZ['SIZE_y']<2.1)]
metro_urban=schoolMergeZ[(schoolMergeZ['METRO']==1)]
metro_sub=schoolMergeZ[(schoolMergeZ['METRO']==2)]
metro_rural=schoolMergeZ[(schoolMergeZ['METRO']==3)]
zmale=schoolMergeZ[(schoolMergeZ['gender']<1.1)]
zfemale=schoolMergeZ[(schoolMergeZ['gender']>1.1)]

#%%
#g = sns.jointplot("total_bill", "tip", data=tips, kind="reg",xlim=(0, 60), ylim=(0, 12), color="r", size=7)

#%%

schoolMergeZ.to_csv('output/fulltable.csv')
correl=schoolMergeZ.corr()
correl.to_csv('output/fullcorrel.csv')

#%%
dict={'large':zlarge,'small':zsmall,'male':zmale,'female':zfemale,'white':zwhite,'black':zblack,
      'urban':metro_urban,'sub':metro_sub,'rural':metro_rural}

for key in dict:
    l=dict[key]
    correl=l.corr()
    #sns.heatmap(correl, cmap='RdYlGn_r',annot=False)   
    path="output/"+key+".csv"
    correl.to_csv(path)


#%% 
#===============
#Partial Regressions
#==============
#maindata = {'large':schoolMergeZ[(schoolMergeZ['SIZE_x']>=150) & (schoolMergeZ['SIZE_x']<2500)],    
maindata = {'large':schoolMergeZ[(schoolMergeZ['SIZE_x']>=250)],
    'small':schoolMergeZ[(schoolMergeZ['SIZE_x']<150)]}

#    'metro_urban':schoolMergeZ[(schoolMergeZ['METRO']==1)],
#    'metro_sub':schoolMergeZ[(schoolMergeZ['METRO']==2)],
#    'metro_rural':schoolMergeZ[(schoolMergeZ['METRO']==3)]}

#maindata['large'][(maindata['large']['white']>0)]
subsets = {'large-white': maindata['large'][(maindata['large']['white']>0)],
           'large-nonwhite': maindata['large'][(maindata['large']['white']<0)],
           'large-male': maindata['large'][(maindata['large']['gender']<0)],
           'large-female': maindata['large'][(maindata['large']['gender']>0)]}

#    'white':schoolMergeZ[(schoolMergeZ['white']>0)],
#    'non-white':schoolMergeZ[(schoolMergeZ['white']<0)],
#    'male':schoolMergeZ[(schoolMergeZ['gender']<1.1)],
#    'female':schoolMergeZ[(schoolMergeZ['gender']>1.1)]}

maindata.update(subsets)

#%%
sum(maindata['large']['SIZE_x']<10)

#%%
#checkign large school and network relationship

ax=sns.lmplot(x='SIZE_x', y="IDGX2", hue="genderrace",data=maindata['large'],fit_reg=True, truncate=True)
ax.set_axis_labels(x_var="size",y_var="in")
ax = plt.gca()
ax.set_title("Size and Ingroup")
sns.jointplot(x='SIZE_x',y='IDGX2',data=maindata['large'],kind='reg') 

#%%
#-----------
#Testing Partial Regression
#-----------

dataset=maindata['large-male']
form = '%s ~ age + black + asian + native + other + %s' % ("IDGX2","agreeableness")
est = smf.ols(formula=form, data=dataset).fit()
fullout = est.summary()
output = fullout.tables[1]
print('large-male')
print(form)
print(fullout)

#%%
#===================
# Multiple Regression
#https://stackoverflow.com/questions/11479064/multiple-linear-regression-in-python

#going to need to do this once we find a good correlation
#===================

import statsmodels.formula.api as smf
from IPython.core.display import HTML

#%%

#What I was imagining was a regression with predictors of
#
#(a) gender,
#
#(b) several variables for race (say, for black, Hispanic, and Asian, with no dummy for white), and then
#
#(c) each of the traits, in a new model one-by-one for each.  The traits would include the big five personality as well as judge-rated personality attractiveness.  
#
# 
#It would help to do all of these one time for small schools and one time for big schools, in case it makes a difference.  It would also help to do all these one time for males and one time for females, and one time for whites and another time for non-whites.  No need for 2-way interactions just yet---they would come into play if we see that it makes a difference when you split the data.
#
# 

#maindata large and small for different school sizes
race=['white','black','asian','other']
variables = ['gender','METRO','age']+race
variables = ['-']
maindeps = ['IDGX2','ODGX2','bonachich']
maininds = ['agreeableness','extraversion','oppenness','concientiousness','neuroticism','personality_attractive']

outfull = open("output\multiregression.txt","w")
outsig = open("output\multiregression_sig.csv","w")
outsig.write(",".join(["significant","schoolsize","networkvar","personalityvar","independentvar","coeff","p","\n"]))

for datatype in ['large']:
    
    dataset = maindata[datatype]
    
    for mainind in maininds:

        
        for maindep in maindeps:

            
            for variable in variables:
            
                form = '%s ~  age + gender + black + asian + other + %s' % (maindep,mainind)
                est = smf.ols(formula=form, data=dataset).fit()
                fullout = est.summary()
                output = fullout.tables[1]
                
                with open('output/temp.csv','w') as w:
                    w.write(output.as_csv())
                csvdf = pd.read_csv('output/temp.csv')
                csvdf = csvdf.iloc[1:]
                p=csvdf['P>|t| ']
                sig = csvdf.where(p<.1).dropna()
                

                if not sig.empty:
                    col1=sig.columns[0]
                    for index, val in sig.iterrows():
                                                
                        pval=val['P>|t| ']
                        coeff=val['   coef   ']
                        signifier=val[col1]
                        
                        outsig.write(",".join([signifier,datatype,maindep,mainind,variable,str(coeff),str(pval)]))
                        outsig.write("\n")
                                                
                            
                               
                outfile=outfull
                outfile.write("\n")    
                outfile.write("=============================\n")
#                outfile.write(maindep + " = " + mainind + " + " + variable  + " + " + mainind  + "*" + variable)
                outfile.write(form)
                outfile.write("\n\n")
                outfile.write("For " + datatype + " schools" )
                outfile.write("\n")
                outfile.write("=============================\n")
                outfile.write(fullout.as_text())
                outfile.write("\n")
                outfile.write("\n")
            
    
outfull.close()
outsig.close()



#%%
with open('output/temp.csv','w') as w:
    w.write(output.as_csv())
csvdf = pd.read_csv('output/temp.csv')
csvdf = csvdf.iloc[1:]

#%%
p=csvdf['P>|t| ']
sig=csvdf.where(p<.1).dropna()

#%%
sig['P>|t| '].iloc[0]
#%%
coeff=sig['   coef   ']                
#%%
for index, val in sig2.iterrows():
    print (index)
    print (val['   coef   '])
    print (val['P>|t| '])

#%% all school plots

#for r in rw4:
#    ax=sns.lmplot(y=r, x="betweenness", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
#    ax.set_axis_labels(x_var="Betweenness", y_var=w4personalitydict[r])
#    ax = plt.gca()
#    ax.set_title("All_Schools")
#    fig = ax.get_figure()
#    figurename = "All_Schools_betweenness_"+w4personalitydict[r].replace("/","_").replace(" ","_")
#    fig.savefig("schoolplots/%s.png"%figurename)
#    
#for r in extraW1dict:
#    print(extraW1dict[r])
#    ax=sns.lmplot(y=r, x="betweenness", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
#    ax.set_axis_labels(x_var="Betweenness", y_var=extraW1dict[r])
#    ax = plt.gca()
#    ax.set_title("All_Schools")
#    figurename = "All_Schools_betweenness_"+extraW1dict[r].replace("/","_").replace(" ","_")
#    fig.savefig("schoolplots/%s.png"%figurename)


#for r in schoolmeta:
#    ax=sns.lmplot(y=r, x="betweenness", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
#    ax.set_axis_labels(x_var="Betweenness", y_var=r)
#    ax = plt.gca()
#    ax.set_title("All_Schools")
#    fig = ax.get_figure()
#    figurename = "All_Schools_betweenness_"+r.replace("/","_").replace(" ","_")
#    fig.savefig("schoolplots/%s.png"%figurename)
#   

#for r in rw4:
#    ax=sns.lmplot(y=r, x="BCENT10X", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
#    ax.set_axis_labels(x_var="Bonachich", y_var=w4personalitydict[r])
#    ax = plt.gca()
#    ax.set_title("All_Schools")
#    fig = ax.get_figure()
#    figurename = "All_Schools_Bonachich_"+w4personalitydict[r].replace("/","_").replace(" ","_")
#    fig.savefig("schoolplots/%s.png"%figurename)
#    
#for r in extraW1dict:
#    print(extraW1dict[r])
#    ax=sns.lmplot(y=r, x="BCENT10X", hue="genderrace",data=schoolMergeRemained,fit_reg=True, truncate=True)
#    ax.set_axis_labels(x_var="Bonachich", y_var=extraW1dict[r])
#    ax = plt.gca()
#    ax.set_title("All_Schools")
#    figurename = "All_Schools_Bonachich_"+extraW1dict[r].replace("/","_").replace(" ","_")
#    fig.savefig("schoolplots/%s.png"%figurename)
#
##ax.set_titles("Test")  # use this argument literally
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

#%%

schoolgroups=schoolMergeRemained.groupby("SCID")
for key,values in schoolgroups:
    school_df = schoolgroups.get_group(key)
#    correl=school_df.corr()
#    #sns.heatmap(correl, cmap='RdYlGn_r',annot=False)
#    print (key)
#    print (correl)
    for r in rw4:
        ax=sns.lmplot(y=r, x="betweenness", hue="genderrace",data=school_df,fit_reg=True, truncate=True)
        ax.set_axis_labels(x_var="Betweenness", y_var=w4personalitydict[r])
        ax = plt.gca()
        ax.set_title("School = " + key)
        fig = ax.get_figure()
        figurename = "School-"+key+"_betweenness_"+w4personalitydict[r].replace("/","_").replace(" ","_")
        fig.savefig("schoolplots/%s.png"%figurename)
        
    for r in extraW1dict:
        print(extraW1dict[r])
        ax=sns.lmplot(y=r, x="betweenness", hue="genderrace",data=school_df,fit_reg=True, truncate=True)
        ax.set_axis_labels(x_var="Betweenness", y_var=extraW1dict[r])
        ax = plt.gca()
        ax.set_title("School = " + key)
        figurename = "School-"+key+"_betweenness_"+extraW1dict[r].replace("/","_").replace(" ","_")
        fig.savefig("schoolplots/%s.png"%figurename)

    for r in rw4:
        ax=sns.lmplot(y=r, x="BCENT10X", hue="genderrace",data=school_df,fit_reg=True, truncate=True)
        ax.set_axis_labels(x_var="Bonachich", y_var=w4personalitydict[r])
        ax = plt.gca()
        ax.set_title("School = " + key)
        fig = ax.get_figure()
        figurename = "School-"+key+"_Bonachich_"+w4personalitydict[r].replace("/","_").replace(" ","_")
        fig.savefig("schoolplots/%s.png"%figurename)
        
    for r in extraW1dict:
        print(extraW1dict[r])
        ax=sns.lmplot(y=r, x="BCENT10X", hue="genderrace",data=school_df,fit_reg=True, truncate=True)
        ax.set_axis_labels(x_var="Bonachich", y_var=extraW1dict[r])
        ax = plt.gca()
        ax.set_title("School = " + key)
        figurename = "School-"+key+"_Bonachich_"+extraW1dict[r].replace("/","_").replace(" ","_")
        fig.savefig("schoolplots/%s.png"%figurename)


# In[16]:
   

schoolNetsmall=schoolNet[['AID']+networkvars]
     
#%%

dna3=pd.read_sas(r"files\DNA\DNA - Wave III\w3dna.xpt",format='xport',encoding='utf-8')
dna4=pd.read_sas(r"files\DNA\DNA - Wave IV\w4dna.xpt",format='xport',encoding='utf-8')

#inhomeWave4
dnamerged=schoolNetsmall.merge(dna3,on='AID',how='left')
dnamerged=dnamerged.merge(dna4,on='AID',how='left')
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
#correl=dnamergedcats.corr()
#sns.heatmap(correl, cmap='RdYlGn_r',annot=False)
##
##slice['variance']=slice.var(axis=1)
##slice['mean']=slice.mean(axis=1)
##slice.describe()
#correl

#%%

from scipy.stats import pearsonr
test=pearsonr([1, 2, 3], [4, 3, 7])

#%%
dnadropped=dnamergedcats.dropna()

#%%
csv = open('output/dnacorrelation.csv','w')
for d in dnamergedcats[networkvars+['betweenness']]:
    csv.write("dna,"+d+",pearson\n")
    for c in dnamergedcats[dnalist]:
        p=pearsonr(dnamergedcats[d],dnamergedcats[c])
        strings = ["%.8f" % x for x in p]
        csv.write(c+","+",".join(strings)+"\n")      
        
csv.close()


#%%
dnamergedcats[c]

#%%
p=pearsonr(dnamergedcats[d],dnamergedcats[c])
strings = ["%.8f" % x for x in p]
csv.write(c+","+",".join(strings)+"/n")


#%%  Experimental %%
# Checking continuity of identical variables
# Need to tie into enumeration-sets to see whether valid-values have changed or not


#schinfo.columns.tolist()
lists=[allWave1.columns.tolist(),allWave2.columns.tolist(),allWave3.columns.tolist(),allWave4.columns.tolist()]

#%%
import re
newlists=[[re.sub("([0-4])", "", i,1) for i in list] for list in lists]

#%%
#a = [1, 2, 3, 4, 5]
#b = [5, 8, 7, 6, 2]

first2 = list(set(newlists[0]) & set(newlists[1]))
second2 = list(set(newlists[0]) & set(newlists[1]))

#this is a list of all variables shared across all waves. 
#would be interesting to see a correlation
#are the results very consistent
#do certain schools or types of people have low consistency
#need to weed out bad values and make sure scales are identical
finallist = list(set(first2) & set(second2))

#surey fatigue and self deception are a major factor in this data

#so maybe something like 
#iterate over the lists
#construct dataframes for each wave
#merge on AID
#filter out nulls
#then correlate