#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 07:35:11 2023
This file Generates Fig 2A, 2B  and Supplementary Figure 4

"""
import numpy as np
import matplotlib.pyplot as plt

import os

import pandas as pd


codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)

import scipy.stats as stats

from decimal import Decimal
import yaml



#%%
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
    axfontSize=18                                                                                                                                       
    isGood=(np.logical_and(np.isfinite(x),np.isfinite(y)))
    #plt.figure(figsize=(10,10))                                                                                                                       
    plt.plot(x[isGood],y[isGood],'ok',markersize=markerSize) 
    plt.tick_params(axis='both', which='major', labelsize=axfontSize)                                                                                         
    plt.xlabel(xlable,fontsize=fontSize,color=xlabelColor)                                                                                                              
    plt.ylabel(ylable,fontsize=fontSize,color=yLabelColor)                                                                                                              
    corr,p=Corr(x,y)                                                                                                    
    #plt.title(titleS+'  Corr:'+str(round(corr,3))+',p-val:'+('%2E' % Decimal(p)),fontsize=20)
    l=plt.legend(title='Corr:'+str(round(corr,2))+'\np-val:'+('%.2E' % Decimal(p)),
                 fontsize=24)
    plt.setp(l.get_title(),fontsize=fontSize)

# %%
figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)

globalParamsFile=os.path.join(homeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))
saveDir=globalParams['Plotting']['saveDir']    
methodColors=globalParams['Plotting']['methodColors']
predName=globalParams['Plotting']['methodNames']['H&E Prediction']
armRenamingDict=globalParams['Plotting']['IMM150']['armNames']
dataDir=figuresParams['Fig2DataDir']

#%% Read IMM150 file

saveFileCsv=os.path.join(dataDir,'All_IMM150.csv')
allRes=pd.read_csv(saveFileCsv)
metricList=['Angio', 'maskAngio', 'pctPos']
allRes['CD31']=[np.float32(s) if is_number(s) else np.NAN for s in allRes['%_CD31']]
allRes=allRes.rename({'TrueAngio':'RNA_Angio'},axis=1)
#%%
exclude_list=globalParams['Data']['IMM150']['failedQcFiles']
duplicate_list= globalParams['Data']['IMM150']['duplicateFiles']
removeList=exclude_list+duplicate_list
genentechRes=allRes[~allRes.FILENAME.isin(removeList)]
genentechRes=genentechRes.reset_index()

#%%TCGA Cohort Cross Validation
fileName=os.path.join(dataDir,'TCGA_CV.csv')
tcgaRes=pd.read_csv(fileName).set_index('SVS')

#%% Pilot cohort predictions.. 
file2=os.path.join(dataDir,'Pilot_res1.csv')
pilotRes=pd.read_csv(file2).set_index('Sample')
#%%  multi cohort plot    Figure 2 A, B
plt.figure(figsize=(30,10))
#plt.subplot(1,3,1)
plt.figure(figsize=(10,10))
plot(tcgaRes['RNA_Angio'],tcgaRes['pctPos'],\
     'RNA Angioscore', predName,'TCGA Holdout',
     xlabelColor=methodColors['RNA Angio'],yLabelColor=methodColors[predName])
saveFile=os.path.join(saveDir,'Fig2a.png')
plt.savefig(saveFile,bbox_inches='tight',dpi=400)
    
#plt.subplot(1,3,2)
plt.figure(figsize=(10,10))
plot(pilotRes['RNA_Angio'],pilotRes['pctPos'],'RNA Angioscore', \
     predName,'Pilot Data',
     xlabelColor=methodColors['RNA Angio'],yLabelColor=methodColors[predName])
saveFile=os.path.join(saveDir,'Fig2b.png')
plt.savefig(saveFile,bbox_inches='tight',dpi=400)

#%%  Supplementary Figure 4
# 2 arm plots, correlation betweeb the arms for Supplementary Figures - TCGA first
# only one plot is used..plot for IMM and Pilot won't be used in the paper
plt.figure(figsize=(10,10))
plot(tcgaRes['RNA_Angio'],tcgaRes['Angio'],\
     'RNA Angioscore', 'Angio Arm Prediction','TCGA Holdout',
     xlabelColor=methodColors['RNA Angio'],yLabelColor=methodColors[predName])
saveFile=os.path.join(saveDir,'FigS4-A1.png')
plt.savefig(saveFile,bbox_inches='tight',dpi=400)
   
#plt.subplot(1,3,3)
plt.figure(figsize=(10,10))
plot(tcgaRes['RNA_Angio'],tcgaRes['pctPos'],\
     'RNA Angioscore', predName,'TCGA Holdout',
     xlabelColor=methodColors['RNA Angio'],yLabelColor=methodColors[predName])
saveFile=os.path.join(saveDir,'FigS4-A2.png')
plt.savefig(saveFile,bbox_inches='tight',dpi=400)  

# plot correlation between the arms
plt.figure(figsize=(10,10))
plot(tcgaRes['Angio'],tcgaRes['pctPos'],\
      'Angio Arm Prediction',predName,'TCGA Holdout',
     xlabelColor=methodColors[predName],yLabelColor=methodColors[predName])
saveFile=os.path.join(saveDir,'FigS4-A3.png')
plt.savefig(saveFile,bbox_inches='tight',dpi=400) 
