#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:18:23 2023
This generates BAP1 PBRM1 WT  Plot for TMAs
Main Figure  3 C
"""

import yaml
import os
import pandas as pd
import numpy as np


codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
 
import matplotlib.pyplot as plt
import seaborn as sns


#%%
figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)
statAnnotDir=figuresParams['statAnnotDir']

sys.path.insert(0, statAnnotDir)
from statannot import add_stat_annotation
#%%
globalParamsFile=os.path.join(homeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))
saveDir=globalParams['Plotting']['saveDir']
methodColors=globalParams['Plotting']['methodColors']
predName=globalParams['Plotting']['methodNames']['H&E Prediction']
pointColor=globalParams['Plotting']['methodColors'][predName]  
saveFile=os.path.join(saveDir,'TMA-Mutation.png')
#%%
yamlFile=figuresParams['Fig3YamlFile']

with open(yamlFile) as file:                                                                                      
    yamlData = yaml.load(file, Loader=yaml.FullLoader)                                                                    



#%%read all data into DataFrames
rawJimDf=pd.read_csv(yamlData['fileJim']) ## 145 values
rawJimDf = rawJimDf.dropna(subset=['SNO']) ## 122 values
rawJimDf['SNO']=rawJimDf['SNO'].astype(int)
j145AngioDf=pd.read_csv(yamlData['jimAngioFile'])
j145AngioDf['pctPos']=100.0*j145AngioDf['pctPos']/(416*416)

idsToRemoveDf=pd.read_csv(yamlData['removeIdFile'])
sno_ToRemove=idsToRemoveDf['J_SNO_Remove'].tolist()
caseId_ToRemove=idsToRemoveDf['V_CaseIds_Remove'].tolist()
# rawVitDf not needed here
rawVitDf=pd.read_csv(yamlData['fileVit'])
vitBAP1Df=pd.read_csv(yamlData['vitBapFile'])
vitBAP1Df=vitBAP1Df[vitBAP1Df['BAP1 IHC']!='x']
vitBAP1Df=vitBAP1Df[~vitBAP1Df['Case identifier'].str.contains("u")]
vitBAP1Df['Case identifier']=vitBAP1Df['Case identifier'].astype(int)
vitalyAngioDf=pd.read_csv(yamlData['vitAngioFile'])
vitalyAngioDf['pctPos']=100.0*vitalyAngioDf['pctPos']/(416*416)
#%% First  set..extract BAP1, PBRM1 info for patinet id..and find avergae angio score
jimScores=[]
sNumbers=[]
mutations=[]

allSno=np.unique(rawJimDf['SNO'])
for snum in allSno:
    if snum not in sno_ToRemove:
        row=j145AngioDf[j145AngioDf['Sno ID']== snum]
        if row.shape[0] > 0:
            jimScores.append(np.nanmean(row['pctPos']))
        else:
            jimScores.append(np.nan)
        rowJim=rawJimDf[rawJimDf['SNO']==snum]
        bap1=rowJim['BAP1_mut'].values[0]
        pbrm1=rowJim['PBRM1_mut'].values[0]
        sNumbers.append(snum)
        print(snum,bap1,pbrm1)
        if bap1 == 1:
            mutations.append('BAP1')
        elif pbrm1 == 1:
            mutations.append('PBRM1')
        else:
            mutations.append('WT')
#%%  now 2nd set
vitScores=[]
vitCaseids=[]
vitMutations=[]

allVitIds=np.unique(vitBAP1Df['Case identifier'])

for caseId in allVitIds:
    if caseId not in caseId_ToRemove:
        row=vitalyAngioDf[vitalyAngioDf['Case_identifier']== caseId]
        if row.shape[0] > 0:
            vitScores.append(np.nanmean(row['pctPos']))
        else:
            vitScores.append(np.nan)
        rowVit=vitBAP1Df[vitBAP1Df['Case identifier']==caseId]
        bap1=int(rowVit['BAP1 IHC'].values[0])
        pbrm1=rowVit['PBRM1 IHC'].values[0]
        print(caseId,bap1,pbrm1)
        if bap1 == 0:
            mutVit='BAP1'
        elif pbrm1 == 0.0:
            mutVit='PBRM1'
        elif bap1 ==1 and pbrm1 == 1.0:
            mutVit='WT'
        else:
            mutVit='UNKNOWN'
        vitCaseids.append(caseId)
        vitMutations.append(mutVit)
#%%
vitFinal=pd.DataFrame({'ID': vitCaseids,'AngioScore': vitScores, 'Mutation': vitMutations})
vitFinal=vitFinal[vitFinal['Mutation']!='UNKNOWN']
jimFinal=pd.DataFrame({'ID': sNumbers,'AngioScore': jimScores, 'Mutation': mutations})
allDf=pd.concat([vitFinal,jimFinal],ignore_index=True)
#%%
allDf=allDf.dropna()
#%% Plot results  Fig. 3C

sns.set(font_scale=1.5)
sns.set_style("white")
fontSize=18

plt.figure(figsize=(10,5))
ax=sns.boxplot(x=allDf['Mutation'],y=allDf['AngioScore'],
               order=['PBRM1','WT','BAP1'], color='lightgray'
               )
sns.swarmplot(x=allDf['Mutation'],y=allDf['AngioScore'],
            order=['PBRM1','WT','BAP1'],size=3,color='k')
add_stat_annotation(ax,x=allDf['Mutation'],y=allDf['AngioScore'],
            order=['PBRM1','WT','BAP1'],
            box_pairs=[('PBRM1','WT'),('WT','BAP1')],
            test='Mann-Whitney',loc='inside', verbose=0,
                                     text_format='simple',pvalue_format_string="{:.2e}",
                                     pvalue_thresholds=[[1e-100,'1e-100']])

plt.ylabel(predName,fontsize=fontSize,color=pointColor)
plt.xlabel('Mutation Status')
saveFile=os.path.join(saveDir,'Fig3c.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)
#plt.title('Effect of Grade on H&E Prediction',fontsize=fontSize)
plt.show()
#%%
   
    

