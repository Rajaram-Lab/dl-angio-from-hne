#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:59:54 2023
Supplementary Figure 5
This code generates the Supplement Figure with 8 images (4 high, 4 low)
along with the pilot correlation plot..the images were manually selected
From pilot data

"""

import os
import pandas as pd
import numpy as np
import yaml


os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

import MixedModel as mm

try: # catch error for new Singularity
   
    import openslide as oSlide
    import NormalizationCore as norm
    
except:
    print('error')
    import openslide as oSlide
    import NormalizationCore as norm

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import matplotlib.colors as colors
from skimage.morphology import remove_small_objects
import cv2

# %% Define Functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
            
def RandSample(patches1,nPixels=10000):
    allPixels=np.reshape(patches1,[-1,1,3])
    if nPixels>0:
        randIdx=np.random.choice(allPixels.shape[0],size=nPixels)
        outPixels=allPixels[randIdx,:]
    else:
        outPixels=allPixels
    return outPixels


def full_path(svsFile,svsDirList):                                                                                                                           
    for svsDir in svsDirList:         
        fpName=os.path.join(svsDir,svsFile)                                                                                                                
        if os.path.exists(fpName):
            return fpName
        else:
            print(svsFile, 'No Full Path Exists', svsDir)
            continue
        
transform =  transforms.Compose([                                                                                                                                                  
        transforms.ToTensor()])  


#%%
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
#%% prepare target data and normalizer

targetFile=figuresParams['FigS5TargetFile']
targetData=np.load(targetFile)

normalizationScheme='Vahadane'
myNormalizer=norm.StainNormalizer(normalizationScheme)
percentileCutoff=95 
#%% X,Y scatter plot data for Pilot Cohort
file2=figuresParams['FigS5File2']

pilotRes=pd.read_csv(file2).set_index('Sample')
x=pilotRes['RNA_Angio'].values
y=pilotRes['Pred_Pct_Pos_MeanE_7'].values

#%%
file1=figuresParams['FigS5File1']

sampleDfAll=pd.read_csv(file1)
#%%only need top 8 data points 
sample8Df=sampleDfAll[:8]
#%% we need Pred Angio for the points so we can plot in red or maroon
allSamplesPred=pilotRes.index.values.tolist()
sampleIds8=sample8Df['Sample'].tolist()
sampleIndices=[allSamplesPred.index(val) for val in sampleIds8]
xsample=[x[i] for i in sampleIndices]
ysample=[y[i] for i in sampleIndices]
isOutlier=np.zeros(pilotRes.shape[0])                                                                                                                                                                            
isOutlier[sampleIndices]=1
#%%
for i in range(sample8Df.shape[0]):
    print(sample8Df.iloc[i]['Sample'], sample8Df.iloc[i]['Pattern'], sample8Df.iloc[i]['Score'],xsample[i], ysample[i])
#%%
#extract images first
svsDirList=figuresParams['FigS5SvsDirList']
fileList=sample8Df['File'].to_list()
#%%
fullPathList=[full_path(svsFile,svsDirList) for svsFile in fileList]                                                                                     
#%% extract 3000x3000 images at 40X from the Files
# they are all DSF=2 or 40X..so no code for accounting dsf
imgList=[]
imgList20X=[]
bigpatchSize=3000
patchSize20X=1500
level=0
patternList=sample8Df['Pattern'].tolist()
trueScores=sample8Df['Score'].tolist()
hneSize=(bigpatchSize,bigpatchSize)
for i in range(sample8Df.shape[0]):
    xval=sample8Df.iloc[i]['X']
    yval=sample8Df.iloc[i]['Y']
    svsFile=sample8Df.iloc[i]['File']
    svsFp=fullPathList[i]
    #svsFp=full_path(svsFile,svsDirList)
    slide=oSlide.open_slide(svsFp)
    slideMpp = np.mean([float(slide.properties[p]) for p in slide.properties if 'mpp' in p.lower()])
    if slideMpp >0.2 and slideMpp<0.4:
        dsf=2
    elif slideMpp>0.4 and slideMpp<0.6:
        dsf=1
    else:
        sys.exit('Invalid dsf')
    cornerPos=(xval,yval)
    
    imgHighRes=np.array(slide.read_region(cornerPos,level,hneSize))[:,:,range(3)]
    img20X=cv2.resize(imgHighRes,(patchSize20X,patchSize20X))
    imgList.append(img20X)
    imgList20X.append(img20X)
    '''
    plt.figure(figsize=(5,5))
    plt.imshow(imgHighRes)
    plt.title(str(i)+' '+svsFile+' X:Y '+str(xval)+' : '+str(yval))
    plt.xticks([],[])
    plt.yticks([],[])
    '''
#%% Normalize Images
img20X_Normalized=[]
#for i in range(1):
for i in range(sample8Df.shape[0]):
    sourceImage=imgList20X[i]
    sourceData=RandSample(sourceImage)
    with suppress_stdout_stderr():
          myNormalizer.fit(np.uint8(targetData),None,np.uint8(sourceData),None,percentileCutoff=95)
    transformedImg=np.squeeze(np.uint8(255*myNormalizer.transformFull(sourceImage)))
    img20X_Normalized.append(transformedImg)
                              
   
# %% Load Model

modelFile=figuresParams['FigS5ModelFile']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=mm.MixedModelSimple(mm.ResNetUNet(2))
model.load_state_dict(torch.load(modelFile))
model=model.to(device)
model.eval()

#%% Predictions
# some predictions are showing some dots..clean up for visualization purpose
cleanSize=400
predictions=[]
predictions_normalized=[]
for i in range(sample8Df.shape[0]):
    
    img=transform(imgList[i])
    img=img.to(device)
    predClasses=torch.argmax(model(img.unsqueeze(0))[0],dim=1)
    predClasses=np.squeeze(predClasses.detach().cpu().numpy())
    toClean=predClasses>0
    cleanpredClasses=remove_small_objects(toClean, min_size=cleanSize,connectivity=2)                                                                                                                                             
    predictions.append(cleanpredClasses)
    
    img=transform(img20X_Normalized[i])
    img=img.to(device)
    predClasses=torch.argmax(model(img.unsqueeze(0))[0],dim=1)
    predClasses=np.squeeze(predClasses.detach().cpu().numpy())
    toClean=predClasses > 0
    cleanpredClasses=remove_small_objects(toClean, min_size=cleanSize,connectivity=2)                                                                                                                                             
    predictions_normalized.append(cleanpredClasses)    


#%% Plot individually and cut and paste them
# plot the scatter
plt.figure(figsize=(8,8))
markerSize=10 
bigSize=12                                                                                                                                    
fontSize=24
isGood=(np.logical_and(np.isfinite(x),np.isfinite(y))) 
plt.plot(x[isGood],y[isGood],'o',color='grey',markersize=markerSize)
#plt.plot(x[isGood],y[isGood],'ok',markersize=markerSize)
Colors=['k','r','purple','orange','yellow','yellowgreen','g','blue']
for i in range(sample8Df.shape[0]):
    plt.plot(xsample[i],ysample[i],'o',color=Colors[i],markersize=bigSize)
 
#plt.scatter(x[isGood],y[isGood],20,cmap=colors.ListedColormap(['lightgray',methodColors[predName]])) 
#plt.scatter(x[isGood],y[isGood],20,c=isOutlier,c=[[0.0,0.0,0.0],methodColors[predName]])
xlable='RNA Angioscore'  
xlabelColor=methodColors['RNA Angio'] 
ylable=predName
yLabelColor=methodColors[predName]                                                                               
plt.xlabel(xlable,fontsize=fontSize,color=xlabelColor)                                                                                                              
plt.ylabel(ylable,fontsize=fontSize,color=yLabelColor)
saveFile=os.path.join(saveDir,'FigS5.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)

#%% fix colormap issues where the overlay makes the H&E image faded
alpha1=0.6 
lineColor1=[0.0,1.0,0.0] # green 
colorsList=[(1,1,1,0),(lineColor1[0],lineColor1[1],lineColor1[2],alpha1)]                                                                                                                                                                          
cmap_green=colors.ListedColormap(colorsList)
#%%Normalized plots
name='FigS5'
for i in range(len(imgList20X)):
    plt.figure(figsize=(9.5,5))
    plt.subplot(1,2,1)
    plt.imshow(imgList20X[i])
    plt.xticks([],[])
    plt.yticks([],[])
    plt.subplot(1,2,2)
    plt.imshow(imgList20X[i])
    #plt.imshow(predictions_normalized[i],vmin=0,vmax=1,cmap=matplotlib.colors.ListedColormap(['w','g']),alpha=0.4)
    plt.imshow(predictions_normalized[i],vmin=0,vmax=1,cmap=cmap_green,alpha=alpha1)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.tight_layout()
    figName=name+str(i)+'.png'
    saveFile=os.path.join(saveDir,figName)
    #plt.savefig(saveFile,bbox_inches='tight',dpi=300)
    

    





    
