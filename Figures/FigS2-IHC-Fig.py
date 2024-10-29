#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:42:23 2023
This works in TensorFlow code

Generates some of the Supplementary Figure 2 on IHS Model Development
Part of the Figure also done using PPT


"""

import yaml

import os as os

import UNet as UNet
import glob 

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
                                                                                                                                                                                            
                                                                                                                 
import pandas as pd                                                                                                                  

import scipy.stats as stats
                                                                                                                                                                     

#%%
codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)

figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)

#%%
globalParamsFile=os.path.join(homeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))

saveDir=globalParams['Plotting']['saveDir']
predName=globalParams['Plotting']['methodNames']['H&E Prediction']
pointColor=globalParams['Plotting']['methodColors'][predName]


#%% below code if for area fraction plot and the 3 figures


marker='CD31'

patchDir=figuresParams['FigS2PatchDir']
annoDir=figuresParams['FigS2PatchDir']
svsDir=figuresParams['FigS2svsDir']

outerBoxClassName='Rectangle'
minBoxSize=[400,400] 
minAnnoBoxArea=0 #Annotations with bounding boxes with area smaller than this are dropped
classDict={'BG':0,'Artifacts':1,marker:2}# Add artifact class here if included

#%%
modelSaveFile=figuresParams['FigS2ModelSaveFile']
unetBaseModel=UNet.LoadUNet(modelSaveFile)
#%%
allPatchFiles=np.array(glob.glob(os.path.join(patchDir,'*.pkl')))
print(allPatchFiles)
all_trainImages=[]
all_trainMasks=[]
for patchfile in allPatchFiles:
    total=0
    trainImages,trainMasks=UNet.LoadPatchesAndMasks([patchfile],classDict)
    for i in range(5):
        all_trainImages.append(trainImages[i])
        all_trainMasks.append(trainMasks[i])
for i, mask in enumerate(all_trainMasks):
    print(all_trainImages[i].shape,mask.shape)

#%%cmposite image..

i=12
ix=int(all_trainMasks[i].shape[0]/2) - 512
iy=int(all_trainMasks[i].shape[1]/2) - 512
image=all_trainImages[i][ix:ix+1024,iy:iy+1024,:]
masked_image=all_trainMasks[i][ix:ix+1024,iy:iy+1024]
mask=np.where(masked_image==2,1,0)
data=np.expand_dims(image/255,0)
prediction=np.squeeze(unetBaseModel.predict(data))
values=np.argmax(prediction,-1)
mask_pred=np.where(values==2,1,0)
#%%read area fraction data data
csvFile=figuresParams['FigS2AreaFracFile']

results=pd.read_csv(csvFile)
true_v=results['true-frac']
pred_v=results['pred-frac']
#%%  Color Map
alpha1=0.6                                                                                                                                            
lineColor1=[0.0,1.0,0.0] # green                                                                                                                      
colorsList=[(1,1,1,0),(lineColor1[0],lineColor1[1],lineColor1[2],alpha1)]
cmap_green=colors.ListedColormap(colorsList) 

#%%  Plot of IHC image
plt.figure(figsize=(6,6))
fontSize=20
markerSize=10
markerColor='green'
plt.imshow(image)
plt.title('CD31 IHC Image',fontsize=fontSize)
plt.xticks([],[])                                                                                                                                                    
plt.yticks([],[])
saveFile=os.path.join(saveDir,'FigS2B.png') 
plt.savefig(saveFile,bbox_inches='tight',dpi=300)
#%% Mask overlay on IHC image
plt.figure(figsize=(6,6))
plt.imshow(image)
plt.imshow(mask,vmin=0,vmax=1,cmap=cmap_green,alpha=alpha1)
frac=round((np.sum(mask)/(mask.shape[0]*mask.shape[1])),3)
plt.title('Annotated Mask Overlay',fontsize=fontSize)
plt.xticks([],[])                                                                                                                                                    
plt.yticks([],[])
saveFile=os.path.join(saveDir,'FigS2C.png') 
plt.savefig(saveFile,bbox_inches='tight',dpi=300)
#%% Virtual IHC overlay on IHC image
plt.figure(figsize=(6,6))
plt.imshow(image)
plt.imshow(mask_pred,vmin=0,vmax=1,cmap=cmap_green, alpha=alpha1)
plt.title('Virtual IHC Overlay',fontsize=fontSize)
plt.xticks([],[])                                                                                                                                                    
plt.yticks([],[])
saveFile=os.path.join(saveDir,'FigS2D.png') 
plt.savefig(saveFile,bbox_inches='tight',dpi=300)
#%% area fraction plot

plt.figure(figsize=(6,6))

m=100*np.minimum(np.min(true_v),np.min(pred_v))
M=100*np.maximum(np.max(true_v),np.max(pred_v))
plt.plot(100*np.array(true_v),100*np.array(pred_v),'ok',color=markerColor,markersize=markerSize)
#plt.plot([m,M],[m,M],'--k')
plt.axis('square')
corr,pVal=stats.pearsonr(true_v,pred_v)
plt.title('Model Performance on Holdout Cohort',fontsize=fontSize)
plt.text(4,25.0,'Corr='+str(np.around(corr,2))+'\np-value='+'{:0.2e}'.format(pVal),
              bbox={'facecolor': 'lightgray', 'alpha': 0.25, 'pad': 10},fontsize=fontSize)
plt.xlabel('CD31 IHC: % Area Fraction',fontsize=fontSize)
plt.ylabel('Virtual CD31 IHC: % Area Fraction',fontsize=fontSize)

plt.tight_layout()

saveFile=os.path.join(saveDir,'FigS2E.png') 
plt.savefig(saveFile,bbox_inches='tight',dpi=300)
