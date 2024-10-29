#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 07:57:34 2023
This File generates Archotectural pattern plot (pattern .vs. Angio Model Prediction )
Supplementary Figure 10
"""
import os as os
import yaml


os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as mplCB

import pandas as pd

import seaborn as sns

# %%
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
 
#%%
resSaveFile=figuresParams['FigS10ResSaveFile']
pyQtDf=pd.read_csv(resSaveFile)
#%% 
patternFile=figuresParams['FigS10PatternFile']
colName='Pattern.Uniformity.2'
newDf=pd.read_excel(patternFile)
#%%
newArch=[]
for i in range(pyQtDf.shape[0]):
    row=pyQtDf.iloc[i]
    fileName=row['SVS']
    sampleId=row['SampleId']
    direction=row['Direction']
    if direction == 'Top':
        matchRow=newDf[(newDf['Sample.ID.original']== sampleId) & (newDf['Image.Top']==fileName)]
    else:
        matchRow=newDf[(newDf['Sample.ID.original']== sampleId) & (newDf['Image.Flip']==fileName)]
    archNew=matchRow[colName].iloc[0]
    newArch.append(archNew.strip())
    print(row['Arch'], archNew)
#%%
pyQtDf['Arch-Simplified']=newArch
#%%
plt.figure(figsize=(10,10))
pyQtDf.boxplot(column='Pred_PctPos',by='Arch-Simplified', grid=False, color='black')
#%%  rename both Small nest, Large Nest as "Nested"
pyQtDf.loc[pyQtDf['Arch-Simplified'] == "Small nest", "Arch-Simplified"] = "Nested"
pyQtDf.loc[pyQtDf['Arch-Simplified'] == "Large nest", "Arch-Simplified"] = "Nested"


#%%  Plot in the new configuration
sns.set(font_scale=2)
sns.set_style("white")
archMap={1:"Nested",2:"Alveolar",3:"Trabecular",
         7:"Solid",8:"Papillary"}

archData=pyQtDf['Arch-Simplified'].values
scores=pyQtDf['Pred_PctPos'].values
fontSize=20
plt.figure(figsize=(15,10))

#plt.title('Relation Between Architecture and Predicted Angioscore ', fontsize=fontSize)
ax=sns.boxplot(x=scores,y=archData, order=["Nested","Alveolar","Trabecular",
         "Solid","Papillary"], color='lightgrey')
plt.xlabel(predName, fontsize=fontSize,color=methodColors[predName])
saveFile=os.path.join(saveDir,'FigS10a.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)
#%% Plot RNA Angio
sns.set(font_scale=2)
sns.set_style("white")
archMap={1:"Nested",2:"Alveolar",3:"Trabecular",
         7:"Solid",8:"Papillary"}

archData=pyQtDf['Arch-Simplified'].values
scores=pyQtDf['RNA_Angio'].values
fontSize=20
plt.figure(figsize=(15,10))

#plt.title('Relation Between Architecture and Predicted Angioscore ', fontsize=fontSize)
ax=sns.boxplot(x=scores,y=archData, order=["Nested","Alveolar","Trabecular",
         "Solid","Papillary"], color='lightgrey')
plt.xlabel('RNA Angio', fontsize=fontSize, color=methodColors['RNA Angio'])
saveFile=os.path.join(saveDir,'FigS10b.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)




