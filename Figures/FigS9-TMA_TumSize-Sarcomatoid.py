#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:58:31 2023
Supplementary Figure 9
This creates Tumorsize and Sarcomatoid plot for the TMAs

"""

import os
import pandas as pd
import numpy as np
import yaml

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import matplotlib.pyplot as plt
import seaborn as sns


#%%
codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)
figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)
    
statAnnotDir=figuresParams['statAnnotationDir']

sys.path.insert(0, statAnnotDir)
from statannot import add_stat_annotation

#%%
globalParamsFile=os.path.join(homeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))

saveDir=globalParams['Plotting']['saveDir']
predName=globalParams['Plotting']['methodNames']['H&E Prediction']
pointColor=globalParams['Plotting']['methodColors'][predName]
#%%
def plot_swarm(xmap, x,y,xtitle, ytitle,plt_title,fontSize,pointColor,saveFile,order_list=['0','1']):

    isGood=np.logical_and(np.isfinite(x),np.isfinite(y))
    newx=x[isGood]
    rhabdoid=[xmap[g] for g in newx]
    plt.figure(figsize=(10,5))
    ax=sns.boxplot(x=rhabdoid,y=y[isGood],color='lightgray', order=order_list)
    sns.swarmplot(x=rhabdoid,y=y[isGood],color='k', order=order_list)
    add_stat_annotation(ax,x=rhabdoid,y=y[isGood],order=order_list,
                box_pairs=[(order_list[0],order_list[1])],test='Mann-Whitney',loc='inside',verbose=0,                                                                                  
                                         text_format='simple',pvalue_format_string="{:.2e}",
                                         pvalue_thresholds=[[1e-100,'1e-100']])
    plt.ylabel(ytitle,fontsize=fontSize,color=pointColor)
    plt.xlabel(xtitle,fontsize=fontSize)  
    #plt.title(plt_title,fontsize=fontSize) 
    plt.savefig(saveFile,bbox_inches='tight',dpi=300) 
    return
#%%

fontSize=18
tumorSizeFile=figuresParams['FigS9TumorSizeFile']
sarcomatoidFile=figuresParams['FigS9SarcomatoidFile']

TumorSizeDf=pd.read_csv(tumorSizeFile)
SarcomatoidDf=pd.read_csv(sarcomatoidFile)
#%%TumorSize plot
# as per Payal's instructions the bin sizes are determined for plotting purpose
TumorSizeDf['sizes']=pd.cut(x=TumorSizeDf['Tum_Size (cm)'],bins=[0, 4, 7, 10, 25],labels=["0-4","4-7","7-10","10-25"])
allScoresTumSize=TumorSizeDf['AngioScore'].tolist()
#%% # Supp Fig S 9 A.
plt.figure(figsize=(8,6))
ax=sns.boxplot(x=TumorSizeDf['sizes'],y=np.array(allScoresTumSize),
               order=["0-4","4-7","7-10","10-25"], color='lightgray')
sns.swarmplot(x=TumorSizeDf['sizes'],y=np.array(allScoresTumSize),color='k',
            order=["0-4","4-7","7-10","10-25"])

add_stat_annotation(ax,x=TumorSizeDf['sizes'],y=np.array(allScoresTumSize),
            order=["0-4","4-7","7-10","10-25"],
            box_pairs=[("0-4","4-7"),("4-7","7-10"),("7-10","10-25")],
            test='Mann-Whitney',loc='inside',verbose=0,                                                                                  
                                     text_format='simple',pvalue_format_string="{:.2e}",
                                     pvalue_thresholds=[[1e-100,'1e-100']])

plt.ylabel(predName,fontsize=fontSize,color=pointColor)
plt.xlabel('Tumor Size (cm)',fontsize=fontSize)
saveFile=os.path.join(saveDir,'FigS9a.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)
#plt.title('Effect of Tumor Size on H&E Prediction',fontsize=fontSize)
#%%Sarcomatoid plot - Supp Fig. 9B

sarcomatoidVals=SarcomatoidDf['Sarcomatoid'].tolist()
allScoresSarcoma=SarcomatoidDf['AngioScore'].tolist()
xmap={1:'1',0:'0'} 
saveFile=os.path.join(saveDir,'FigS9b.png')                                                                                                                       
plot_swarm(xmap,np.array(sarcomatoidVals),np.array(allScoresSarcoma),'Sarcomatoid Status',predName,'Effect of Sarcomatoid Status on H&E Prediction',fontSize,
           pointColor,saveFile)

#plt.savefig(saveFile,bbox_inches='tight')
#%%
 
medians = SarcomatoidDf.groupby(['Sarcomatoid'])['AngioScore'].median()
