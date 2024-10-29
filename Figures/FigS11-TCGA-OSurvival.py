#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:52:31 2023
Code for TCGA_Survival Analysis - Supplementary Fig. 11

"""
import yaml
import os
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

import matplotlib.pyplot as plt


from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

# %% Load up Various Params
codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)
figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)

globalParamsFile=os.path.join(homeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))

saveDir=globalParams['Plotting']['saveDir']  
methodColors=globalParams['Plotting']['methodColors']
predName=globalParams['Plotting']['methodNames']['H&E Prediction']
armRenamingDict=globalParams['Plotting']['IMM150']['armNames']

#%% Extract TCGA Long IDs and add to Raw daThis is for survival analysis of all TCGA data (combined train and test)

angioResultsFile=figuresParams['FigS11AngioResFile']
resDf=pd.read_csv(angioResultsFile)
column1='Mixed_Angio_F0_E7'
column2='Mixed_MaskAngio_F0_E7'
column3='PctPos_F0_E7'
#%%
# lets get LONG IDS for Fold number =0
foldNum=0
friendlyNameMappingFile=figuresParams['FigS11FriendlyMappingFile']   
  
friendlyNameMapping=pd.read_csv(friendlyNameMappingFile)                                                                               
friendlyNames=[os.path.split(s)[-1].replace('.svs','.hdf5')                                                                            
               for s in friendlyNameMapping['Friendly Path'].values]                                                                   
tcgaIds=['-'.join(os.path.split(s)[-1].split('-')[:3])                                                                                 
               for s in friendlyNameMapping['Original Path'].values]                                                                   
tcgaLongIds=[('-'.join(os.path.split(s)[-1].split('-')[:4])[:-1])                                                                      
               for s in friendlyNameMapping['Original Path'].values]


#%%
# find the longID corresponsing to SVS name
longids=[]
SVSNames=resDf['SVS'].tolist()
for indx, name in enumerate(SVSNames):
    loc=friendlyNames.index(name)
    idval=tcgaLongIds[loc]
    longids.append(idval)
    print(name, loc, idval)
#%%
tcga_angio_Df=pd.DataFrame({'TestTcgaLongIds': longids,'True_RNA':resDf['TestScores'].values,'Angio_RNA': resDf[column1].values,
                            'Mask_Angio': resDf[column2].values,'pctPos': resDf[column3].values})

#%%#Clean up rows with no Survival info
survivalFile=figuresParams['FigS11TCGASurvivalFile']
survivalData=pd.read_csv(survivalFile,skiprows=4,delimiter='\t')
time=[]
is_observed=[]
Longidvals=tcga_angio_Df['TestTcgaLongIds'].tolist()
idvals=[val[:-3] for val in Longidvals]
#idvals=tcga_angio_Df['TestSampleIds'].tolist()
survivalIds=survivalData['PATIENT_ID'].tolist()
ids_Nodata=[]
for ix,idval in enumerate(idvals):
    if idval in survivalIds:
        row=survivalData[survivalData['PATIENT_ID']==idval]
        timeVal=row['OS_MONTHS'].values[0]
        observed=row['OS_STATUS'].values[0]
        time.append(timeVal/12.0)
        is_observed.append(int(observed.split(':')[0]))
        print(idval, timeVal, observed)
    else:
        print('no survival info for id', idval)
        ids_Nodata.append(Longidvals[ix])
#%%remove all rows with ids_Nodata

tcga_angio_Df=tcga_angio_Df[~tcga_angio_Df.TestTcgaLongIds.isin(ids_Nodata)]
#%%
patientAngio=tcga_angio_Df['pctPos'].values
patientIsObserved=np.array(is_observed)
patientDuration=  np.array(time)

patientIsObserved=patientIsObserved[np.isfinite(patientAngio)]
patientDuration=patientDuration[np.isfinite(patientAngio)]
patientAngio=patientAngio[np.isfinite(patientAngio)]

ci=concordance_index(patientDuration,patientAngio,patientIsObserved)
print('Cindex :', round(ci,3),'  Median:', np.median(patientAngio))

threshvals=np.percentile(patientAngio[np.isfinite(patientAngio)],np.linspace(10,80,20))
pVals=np.zeros(len(threshvals))

for threshCounter,angioThresh in enumerate(threshvals):
#cd31Thresh=0.014
   
    isLow=patientAngio<angioThresh
    isHigh=patientAngio>=angioThresh

    
    results = logrank_test(patientDuration[isLow], patientDuration[isHigh],
                               event_observed_A=patientIsObserved[isLow], 
                               event_observed_B=patientIsObserved[isHigh])
    #print(results.summary)
            
        #  Calculate effect size
    dfLow=pd.DataFrame({'deceased_1':patientIsObserved[isLow],
             'ttd days':patientDuration[isLow],
             'isAngioLow':np.ones(np.sum(isLow))})
    dfHigh=pd.DataFrame({'deceased_1':patientIsObserved[isHigh],
             'ttd days':patientDuration[isHigh],
             'isAngioLow':np.zeros(np.sum(isHigh))})
    
    coxDf=pd.concat([dfLow,dfHigh])
    cph = CoxPHFitter()
    cph.fit(coxDf, duration_col='ttd days', event_col='deceased_1')
    pVals[threshCounter]=cph.summary.loc['isAngioLow']['p']
    #print(cph.summary.loc['isCd31Low'])
    print(round(angioThresh,3),round(cph.summary.loc['isAngioLow']['exp(coef)'],3),round(cph.summary.loc['isAngioLow']['p'],9),
          round(cph.summary.loc['isAngioLow']['exp(coef) lower 95%'],3),round(cph.summary.loc['isAngioLow']['exp(coef) upper 95%'],3),
          len(dfLow), len(dfHigh))
    
# %%  Supplementary Fig. 11

plt.plot(threshvals,-np.log10(pVals),'ok-')
#plt.xlabel('H&E DL Angioscore',fontsize=16)
plt.xlabel(predName,fontsize=16,color=methodColors[predName])
plt.ylabel('$-log_{10}(p)$',fontsize=16)
#plt.title('TCGA',fontsize=16)
ylim=plt.ylim()
plt.vlines(5.66,ylim[0],ylim[1],'darkred')
plt.vlines(10.37,ylim[0],ylim[1],'darkred')
plt.text(4,6.5,'Low\nAngio',ha='center',fontsize=14) 
plt.text((5.66+10.37)/2,6.5,'Med\nAngio',ha='center',fontsize=14)  
plt.text(13,6.5,'High\nAngio',ha='center',fontsize=14)

saveFile=os.path.join(saveDir,'TCGA_CutOff.png')                                                                                
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)  
