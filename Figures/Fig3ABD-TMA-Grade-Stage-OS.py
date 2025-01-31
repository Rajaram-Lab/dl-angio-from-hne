#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:10:45 2023
This generates Grade, Stage, Overall Survival Plots for TMAs
Main Figure  3 A,B, and D

"""

import os
import pandas as pd
import numpy as np

from configparser import ConfigParser
config = ConfigParser()
config.read(os.path.join( os.getenv("HOME"),'.pythonConfig.ini'))

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(0, config.get('DL', 'dlCoreDir'))

import matplotlib.pyplot as plt
import seaborn as sns



from lifelines import KaplanMeierFitter,CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import yaml
# %%
codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)


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
predName=globalParams['Plotting']['methodNames']['H&E Prediction'] 
pointColor=globalParams['Plotting']['methodColors'][predName]  
#%% Read Data

GradesFile=figuresParams['Fig3GradesFile']
StageFile=figuresParams['Fig3StageFile']
SurvivFile=figuresParams['Fig3SurvivFile']
GradeDf=pd.read_csv(GradesFile)
StageDf=pd.read_csv(StageFile)
SurvDf=pd.read_csv(SurvivFile)

   # to be read from Some common file
#%%Grade Plot - Fig. 3A
sns.set(font_scale=1.5)
sns.set_style("white")
gradeData=GradeDf['Grade'].tolist()
allScoresGrade=GradeDf['AngioScore'].tolist()
gradeMap={1:'Grade 1',2:'Grade 2',3:'Grade 3',4:'Grade 4'} 
fontSize=18
grades=[gradeMap[g] for g in gradeData]
plt.figure(figsize=(10,5))
ax=sns.boxplot(x=grades,y=np.array(allScoresGrade),
               order=['Grade 1','Grade 2','Grade 3', 'Grade 4'], color='lightgray'
               )
sns.swarmplot(x=grades,y=np.array(allScoresGrade),
            order=['Grade 1','Grade 2','Grade 3', 'Grade 4'],size=3,color='k')
add_stat_annotation(ax,x=grades,y=np.array(allScoresGrade),
            order=['Grade 1','Grade 2','Grade 3', 'Grade 4'],
            box_pairs=[('Grade 1','Grade 2'),('Grade 2', 'Grade 3'),('Grade 3', 'Grade 4')],
            test='Mann-Whitney',loc='inside', verbose=0,
                                     text_format='simple',pvalue_format_string="{:.2e}",
                                     pvalue_thresholds=[[1e-100,'1e-100']])

plt.ylabel(predName,fontsize=fontSize,color=pointColor)
saveFile=os.path.join(saveDir,'Fig3a.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)
#plt.title('Effect of Grade on H&E Prediction',fontsize=fontSize)
plt.show()
#%% Stage Plot.  Fig. 3B
c_stages = StageDf['Stage'].tolist()                                                                                                                 
c_Yvals= StageDf['AngioScore'].tolist()
plt.figure(figsize=(10,5))                                                                                                                
ax=sns.boxplot(x=c_stages,y=c_Yvals,order=['Low Stage','High Stage'], 
               color='lightgray') 
sns.swarmplot(x=c_stages,y=c_Yvals,                                                                                            
            order=['Low Stage','High Stage'],size=3,color='k')                                                                                             
add_stat_annotation(ax,x=c_stages,y=c_Yvals,                                                                                              
            order=['Low Stage','High Stage'],                                                                                             
            box_pairs=[('Low Stage','High Stage')],                                                                                       
            test='Mann-Whitney',loc='inside', verbose=0,                                                                                  
                                     text_format='simple',pvalue_format_string="{:.2e}",
                                     pvalue_thresholds=[[1e-100,'1e-100']])                                                                                  
plt.ylabel(predName,fontsize=fontSize,color=pointColor) 
saveFile=os.path.join(saveDir,'Fig3b.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)                                                                                             
#plt.title('Effect of Stage on H&E Prediction',fontsize=fontSize)
plt.show()
#%% Survival Plot.  There are some Nans for Angio Scores (no punches)..
# dropna results in Data for 520 patients
SurvDf=SurvDf.dropna()

survTime=SurvDf['Time'].tolist()
survEvent=SurvDf['Event'].tolist()
allScoresSurv=SurvDf['AngioScore']

patientAngio=np.zeros(len(survEvent))
patientIsObserved=np.zeros(len(survEvent))
patientDuration=np.zeros(len(survEvent))

patientAngio=np.array(allScoresSurv)
patientIsObserved=np.array(survEvent)
patientDuration=  np.array(survTime)

#   two threshold plot

ci=concordance_index(patientDuration,patientAngio,patientIsObserved)
print('Cindex :', round(ci,3),'  Median:', np.median(patientAngio))
angioThresh1=10.37

isLow=patientAngio<angioThresh1
isHigh=patientAngio>=angioThresh1

results = logrank_test(patientDuration[isLow], patientDuration[isHigh],
                           event_observed_A=patientIsObserved[isLow], 
                           event_observed_B=patientIsObserved[isHigh])
#print(results.summary)
        
    #  Calculate effect size
dfRest=pd.DataFrame({'deceased_1':patientIsObserved[isLow],
         'ttd days':patientDuration[isLow],
         'isAngioLow':np.ones(np.sum(isLow))})
dfHigh=pd.DataFrame({'deceased_1':patientIsObserved[isHigh],
         'ttd days':patientDuration[isHigh],
         'isAngioLow':np.zeros(np.sum(isHigh))})

angioThresh2=5.66
isLowest=patientAngio<angioThresh2

dfLow=pd.DataFrame({'deceased_1':patientIsObserved[isLowest],
         'ttd days':patientDuration[isLowest],
         'isAngioLow':np.ones(np.sum(isLowest))})

isMiddle=np.logical_and((patientAngio<=angioThresh1) , (patientAngio>=angioThresh2))

dfMid = pd.DataFrame({'deceased_1':patientIsObserved[isMiddle],
         'ttd days':patientDuration[isMiddle],
         'isAngioLow':np.zeros(np.sum(isMiddle))})


#%% Fig. 3D
plt.figure(figsize=(11,11))
fontSize=20
fontSize1=24
axfontSize=18
kmf = KaplanMeierFitter()
low_num=np.sum(isLow)
high_num=np.sum(isHigh)
kmf.fit(dfHigh['ttd days'],dfHigh['deceased_1'],label=(('High Angio(N=')+str(high_num)+')'))
ax=kmf.plot(ci_show=False,color=pointColor,
            linestyle=globalParams['Plotting']['highMedLowLineStyles']['High'],
            linewidth=3)
kmf.fit(dfMid['ttd days'],dfMid['deceased_1'],label=(('Medium Angio(N=')+str(dfMid.shape[0])+')'))
ax=kmf.plot(ci_show=False,color=pointColor,
            linestyle=globalParams['Plotting']['highMedLowLineStyles']['Med'],
            linewidth=3)
kmf.fit(dfLow['ttd days'],dfLow['deceased_1'],label=(('Low Angio(N=')+str(dfLow.shape[0])+')'))
ax=kmf.plot(ci_show=False,color=pointColor,
            linestyle=globalParams['Plotting']['highMedLowLineStyles']['Low'],
            linewidth=3)
plt.tick_params(axis='both', which='major', labelsize=axfontSize)
plt.xlabel('Time (Years)',fontsize=fontSize1)
plt.ylabel('Overall Survival Probability',fontsize=fontSize1)

plt.legend(prop={'size': fontSize})
saveFile=os.path.join(saveDir,'Fig3d.png')
plt.savefig(saveFile,bbox_inches='tight',dpi=400)
plt.show()
#%%get HR and pVal for two groups Need to put these values into the above plot once we determine how to do it
#High .vs. low

dfHighLow=pd.concat([dfHigh, dfLow],axis=0)
dfHighLow['isAngioLow']=dfHighLow['isAngioLow'].astype(int)
isLow=dfHighLow['isAngioLow'].values
cph = CoxPHFitter()
cph.fit(dfHighLow, duration_col='ttd days', event_col='deceased_1')

print('High - Low',round(cph.summary.loc['isAngioLow']['exp(coef)'],3),round(cph.summary.loc['isAngioLow']['p'],9),
      round(cph.summary.loc['isAngioLow']['exp(coef) lower 95%'],3),round(cph.summary.loc['isAngioLow']['exp(coef) upper 95%'],3),
      str(np.sum(isLow)), str(np.sum(isLow==0)))
#%%Mid .vs Low
dfMidLow=pd.concat([dfMid, dfLow],axis=0)
dfMidLow['isAngioLow']=dfMidLow['isAngioLow'].astype(int)
isLow=dfMidLow['isAngioLow'].values
cph1 = CoxPHFitter()
cph1.fit(dfMidLow, duration_col='ttd days', event_col='deceased_1')

print('Med - Low',round(cph1.summary.loc['isAngioLow']['exp(coef)'],3),round(cph1.summary.loc['isAngioLow']['p'],9),
      round(cph1.summary.loc['isAngioLow']['exp(coef) lower 95%'],3),round(cph1.summary.loc['isAngioLow']['exp(coef) upper 95%'],3),
      str(np.sum(isLow)), str(np.sum(isLow==0)))

