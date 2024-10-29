#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 10:15:40 2021
Plots Supplementary Figure 7
CD31 plots related to Pilot Phase 1 Data

"""
import os as os
import yaml
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from decimal import Decimal

#%%  plot results

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def Corr(x,y,isGood=None,measure='spearman'):                                                                                                            
    if isGood is None:                                                                                                                
        isGood=np.logical_and(np.isfinite(x),np.isfinite(y))  
    assert measure in ['spearman','pearson']
    if measure == 'spearman':                                                      
        corr,p=stats.spearmanr(x[isGood],y[isGood])  
    else:    
        corr,p=stats.pearsonr(x[isGood],y[isGood])                                                                               
    return corr,p     

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True


    
def plot(x,y,xlable, ylable, titleS,xlabelColor,yLabelColor):                                                                                                                 
    markerSize=10                                                                                                                                     
    fontSize=24                                                                                                                                       
    isGood=(np.logical_and(np.isfinite(x),np.isfinite(y)))                                                                                            
    #plt.figure(figsize=(10,10))                                                                                                                       
    plt.plot(x[isGood],y[isGood],'ok',markersize=markerSize)                                                                                          
    plt.xlabel(xlable,fontsize=fontSize,color=xlabelColor)                                                                                                              
    plt.ylabel(ylable,fontsize=fontSize,color=yLabelColor)                                                                                                              
    corr,p=Corr(x,y)                                                                                                    
    #plt.title(titleS+'  Corr:'+str(round(corr,3))+',p-val:'+('%2E' % Decimal(p)),fontsize=20)
    l=plt.legend(title='Corr:'+str(round(corr,2))+'\np-val:'+('%.2E' % Decimal(p)),
                 fontsize=24)
    plt.setp(l.get_title(),fontsize=fontSize)
# %% read global params
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
#%% this has the results on CD31 IHC predictions
dataFile=figuresParams['FigS7DataFile']
dataDf=pd.read_csv(dataFile)

#%%
file1=figuresParams['FigS7File1']

allDf=pd.read_csv(file1)

#%%
sampleList=dataDf['rnaSampleNames'].to_list()
#%%
# only get pilot data from Phase 1
miniDf=allDf[allDf['Sample'].isin(sampleList)]
#%%
dataDf=dataDf.rename({'rnaSampleNames':'Sample'}, axis=1)
#%%
bigDf1=dataDf.merge(miniDf, on='Sample')

#%%

plt.figure(figsize=(10,10))
plot(bigDf1['ihcPct'],bigDf1['RNA_Angio'],\
     'CD31', 'RNA Angioscore','UTSEQ-CD31',
     xlabelColor=methodColors['CD31'],yLabelColor=methodColors['RNA Angio'])
saveFile=os.path.join(saveDir,'FigS7-A.png')
#plt.savefig(saveFile,bbox_inches='tight', dpi=300)
#%%
plt.figure(figsize=(10,10))
plot(bigDf1['Pred_Pct_Pos_FlipE_7'],bigDf1['ihcPct'],\
     predName, 'CD31','UTSEQ-CD31',
     xlabelColor=methodColors[predName],yLabelColor=methodColors['CD31'])
saveFile=os.path.join(saveDir,'FigS7-B.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)
#%%
plt.figure(figsize=(10,10))
plot(bigDf1['RNA_Angio'],bigDf1['Pred_Pct_Pos_FlipE_7'],\
     'RNA Angioscore',predName,'UTSEQ-CD31',
     xlabelColor=methodColors['RNA Angio'],yLabelColor=methodColors[predName])
saveFile=os.path.join(saveDir,'FigS7-C.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)

