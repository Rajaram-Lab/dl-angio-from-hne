#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:37:56 2023
This is the TKI resPonse curve Plot for UTSW data - Figure 4A
"""

import os
import pandas as pd
import numpy as np
import yaml
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter,CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
#%%

codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)

figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)

#%%
globalParamsFile=os.path.join(homeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))
predName=globalParams['Plotting']['methodNames']['H&E Prediction']
pointColor=globalParams['Plotting']['methodColors'][predName]
saveDir=globalParams['Plotting']['saveDir']

#%%
saveFile=figuresParams['Fig4saveFile']

tkiDf=pd.read_csv(saveFile)
#%%create survDF after filtering
field='Pred_Pos'
isGood=np.logical_not(pd.isnull(tkiDf[field].values))
isStopValid=tkiDf['Stop_Reason'].values != 'Toxicity' 
isGood=np.logical_and(isGood,isStopValid)
SurvDf=tkiDf[isGood]

#%%
angioThresh=5.66
survTime=SurvDf['TNT(years)'].tolist()
survEvent=SurvDf['isObserved'].tolist()
allScoresSurv=SurvDf['Pred_Pos']

patientAngio=np.zeros(len(survEvent))
patientIsObserved=np.zeros(len(survEvent))
patientDuration=np.zeros(len(survEvent))

patientAngio=np.array(allScoresSurv)
patientIsObserved=np.array(survEvent)
patientDuration=  np.array(survTime)

isLow=patientAngio<angioThresh
isHigh=patientAngio>=angioThresh

dfHigh=pd.DataFrame({'isObserved':patientIsObserved[isHigh],
         'TNT(years)':patientDuration[isHigh],
         'isAngioLow':np.zeros(np.sum(isHigh))})

dfLow=pd.DataFrame({'isObserved':patientIsObserved[isLow],
         'TNT(years)':patientDuration[isLow],
         'isAngioLow':np.ones(np.sum(isLow))})
#%%
plt.figure(figsize=(8,8))
fontSize=18
kmf = KaplanMeierFitter()
low_num=np.sum(isLow)
high_num=np.sum(isHigh)
kmf.fit(dfHigh['TNT(years)'],dfHigh['isObserved'],label=(('High Angio(N=')+str(high_num)+')'))
ax=kmf.plot(ci_show=False,color=pointColor,
            linestyle=globalParams['Plotting']['highMedLowLineStyles']['High'],
            linewidth=3)

kmf.fit(dfLow['TNT(years)'],dfLow['isObserved'],label=(('Low Angio(N=')+str(low_num)+')'))
ax=kmf.plot(ci_show=False,color=pointColor,
            linestyle=globalParams['Plotting']['highMedLowLineStyles']['Low'],
            linewidth=3)
plt.xlabel('Time (Years)',fontsize=fontSize)
plt.ylabel('Time to Next Treatment Probability',fontsize=fontSize)
#plt.title('UTSW Cohort: Time-To-Next-Treatment Response Curves',fontsize=fontSize)

plt.legend(prop={'size': fontSize})
saveFile=os.path.join(saveDir,'Fig4A.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)
plt.show()
#%%
dfHighLow=pd.concat([dfHigh, dfLow],axis=0)
dfHighLow['isAngioLow']=dfHighLow['isAngioLow'].astype(int)
isLow=dfHighLow['isAngioLow'].values
cph = CoxPHFitter()
cph.fit(dfHighLow, duration_col='TNT(years)', event_col='isObserved')

ci=concordance_index(SurvDf['TNT(years)'],SurvDf['Pred_Pos'].values,SurvDf['isObserved'])

print('Ci : ', ci, 'High - Low',round(cph.summary.loc['isAngioLow']['exp(coef)'],3),round(cph.summary.loc['isAngioLow']['p'],9),
      round(cph.summary.loc['isAngioLow']['exp(coef) lower 95%'],3),round(cph.summary.loc['isAngioLow']['exp(coef) upper 95%'],3),
      str(np.sum(isLow)), str(np.sum(isLow==0)))

