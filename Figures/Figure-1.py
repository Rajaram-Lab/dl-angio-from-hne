#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 07:47:19 2023
Most of Figure 1 generated in PPT.
This File Generates a small portion of Figure 1C
"""

import glob
import os
import pandas as pd
import numpy as np

import yaml

os.environ["CUDA_VISIBLE_DEVICES"]="0"

codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)
import sys
try: # catch error for new Singularity
   
    import openslide as oSlide
    import NormalizationCore as norm
    

except:
    print('error')
    import openslide as oSlide
    import NormalizationCore as norm

import pickle as pkl
import matplotlib.pyplot as plt

#import seaborn as sns
import scipy.sparse


import matplotlib.colors as colors



import cv2
from skimage.measure import block_reduce
from statsmodels.distributions.empirical_distribution import ECDF

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
def RandSample(patches,nPixels=10000):
    allPixels=np.reshape(patches,[-1,1,3])
    if nPixels>0:
        randIdx=np.random.choice(allPixels.shape[0],size=nPixels)
        outPixels=allPixels[randIdx,:]
    else:
        outPixels=allPixels
    return outPixels
#%%
figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)

globalParamsFile=os.path.join(homeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))
pltSaveDir=globalParams['Plotting']['saveDir']    
methodColors=globalParams['Plotting']['methodColors']
predName=globalParams['Plotting']['methodNames']['H&E Prediction'] 
#%%
#  use the second file 
svsFiles=figuresParams['Fig1SvsFiles']
saveDir=figuresParams['Fig1saveDir']
maskDir=figuresParams['Fig1maskDir']
svsDir=figuresParams['Fig1svsDir']

svsList=[os.path.join(svsDir,fileName) for fileName in svsFiles]
tumorMaskFileList=[]
cd31MaskFileList=[os.path.join(saveDir,os.path.split(svsName)[-1].replace('.svs','-Normalized.npz')) for svsName in svsList ]
tumorMaskFileList=[os.path.join(maskDir,os.path.split(svsName)[-1].replace('.svs','.pkl')) for svsName in svsList]
heatMapFileList=[os.path.join(saveDir,svsName.replace('.svs','-Normalized.pkl')) for svsName in svsFiles]
#%%
patternsAll=[['Small Nest','Small Nest','Trabecular','Solid'],
             ['Small Nest','Trabecular','Small Nest','Trabecular']
             ]
rna_angioAll=[[7.3,6.1,5.3,'Not Available'],
              [6.74,5.93,6.82,5.93]
              ]
cornerPosListAll=[[(62240,24000),(86920,39300),(28890,47420),(64350,72630)],
                  [(43526,27360),(70101,49177),(38110,50748),(68328,53735)]
                  ]

#%%
indx=1
slideMagLevel=3
svsFile=svsList[indx]
maskSaveFile=cd31MaskFileList[indx]
maskFile=tumorMaskFileList[indx]
mask=pkl.load(open(maskFile,'rb'))[0]
slide=oSlide.open_slide(svsFile)
predClasses=np.array(np.uint8(scipy.sparse.load_npz(maskSaveFile).todense()))
maskResized=cv2.resize(mask,predClasses.shape[::-1])
level=min(slide.level_count-1,slideMagLevel)
imgLowRes=np.array(slide.read_region((0,0),level,slide.level_dimensions[level]))[:,:,range(3)]

fracPosResized=pkl.load(open(heatMapFileList[indx],'rb'))
#%%
patterns=patternsAll[indx]
rna_angio = rna_angioAll[indx]
cornerPosList = cornerPosListAll[indx]

imgList=[]
maskList=[]
predPos=[]
bigpatchSize=2000
patchSize20X=1000
level=0
hneSize=(bigpatchSize,bigpatchSize)
for i in range(4):
    cornerPos=cornerPosList[i]
    maskcornerPos=(int(cornerPos[0]/2),int(cornerPos[1]/2))
    imgHighRes=np.array(slide.read_region(cornerPos,level,hneSize))[:,:,range(3)]
    img20x=cv2.resize(imgHighRes,(patchSize20X,patchSize20X))
    maskHighRes=predClasses[maskcornerPos[1]:maskcornerPos[1]+patchSize20X,
                     maskcornerPos[0]: maskcornerPos[0]+patchSize20X]
    imgList.append(img20x)
    maskList.append(maskHighRes)
    pred=np.round(100.0*(np.sum(maskHighRes)/(patchSize20X*patchSize20X)),2)
    predPos.append(pred)
#%%
# normalize the image list to brighter colors for presentation purpose
normalizationScheme='Vahadane'
myNormalizer=norm.StainNormalizer(normalizationScheme)
percentileCutoff=95
# change the target from TCGA sample to specific sample from slide

targetSlide=figuresParams['Fig1TargetSlide']
indx=0
bigpatchSize=1000
level=0
hneSize=(bigpatchSize,bigpatchSize)
cornerPos=(23363,15255)
slide_target=oSlide.open_slide(targetSlide)
targetImage=np.array(slide_target.read_region(cornerPos,level,hneSize))[:,:,range(3)]
targetData=RandSample(targetImage)
nimgList=[]
for image_indx in range(4):
    sourceImage=imgList[image_indx]
    sourceData=RandSample(sourceImage)
    with suppress_stdout_stderr():
        myNormalizer.fit(np.uint8(targetData),None,np.uint8(sourceData),None,percentileCutoff=95)
    normImage=np.uint8(255*myNormalizer.transformFull(np.uint8(sourceImage)))
    nimgList.append(normImage)

#%%  adress shading issues
alpha1=0.6 
lineColor1=[0.0,1.0,0.0] # green 
colorsList=[(1,1,1,0),(lineColor1[0],lineColor1[1],lineColor1[2],alpha1)]                                                                                                                                                                          
cmap_green=colors.ListedColormap(colorsList)

# %%

predClassesResized=cv2.resize(predClasses,(imgLowRes.shape[1],imgLowRes.shape[0]))

maskResized1=cv2.resize(mask,predClassesResized.shape[::-1])
predClassesResized[np.logical_not(maskResized1)]=0
# %%
isFg=cv2.resize(np.uint8(np.any(imgLowRes<220,axis=-1)),predClasses.shape[::-1])
predClassesTumor=np.logical_and(predClasses,np.logical_and(maskResized,isFg))

blockSize=416
nPos=block_reduce(predClassesTumor,(blockSize,blockSize),func=np.sum)
nTum=block_reduce(maskResized,(blockSize,blockSize),func=np.sum)

fracPos=nPos/nTum
fracPos[nTum/(blockSize*blockSize)<0.95]=np.NAN
fracPosResized=cv2.resize(fracPos,(imgLowRes.shape[1],imgLowRes.shape[0]),interpolation=cv2.INTER_NEAREST)


# %%
# this draws the Percentile Color of Rings this needs to be run for ecdf variable
file2=figuresParams['Fig1File2']
pilotRes=pd.read_csv(file2).set_index('Sample')

import matplotlib
cmapName='hot_r'
cmap = matplotlib.cm.get_cmap(cmapName)



ecdf=ECDF(pilotRes['RNA_Angio'].values)
temp=[]
temp0=[]
for i in range(2):
    for x in rna_angioAll[i]:
        if not isinstance(x, str):
            print(x,ecdf(x),np.uint8(np.array(list(cmap(ecdf(x))))*255)[:-1])
            #plt.scatter(i,x,100,c=np.array(list(cmap(ecdf(x)))),cmap=cmap)
            temp.append(x)
            temp0.append(i)
plt.scatter(temp0,temp,100,ecdf(temp),vmin=0,vmax=1,cmap=cmapName)
plt.ylabel('RNA Angio')
plt.xlabel('Sample')
plt.title('Percentile Coloring of Rings')
plt.xticks([],[])
plt.colorbar()

# %%  This draws the Central image of slide with 3 Rings
punchCenters=[(43526, 27360), (68328+2000, 53735+2000), (38110, 50748)]
plt.figure(figsize=(10,10))
ax=plt.gca()
plt.imshow(imgLowRes)
plt.imshow(fracPosResized,cmap='bwr',vmin=0,vmax=0.3)

for i,x in enumerate(rna_angioAll[1][:3]):
    if not isinstance(x, str):
        print(x,ecdf(x),np.uint8(np.array(list(cmap(ecdf(x))))*255)[:-1])
        temp.append(x)
        temp0.append(i)
        c=plt.Circle(tuple(np.array(punchCenters[i])/16),radius=200,
                     edgecolor=np.array(list(cmap(ecdf(x)))),facecolor=(0,0,0,0),
                     linewidth=7)        
        ax.add_patch(c)
plt.vlines(np.arange(0,imgLowRes.shape[1],blockSize/8), -0.5, imgLowRes.shape[0]+0.5,linewidth=0.2,color='k')
plt.hlines(np.arange(0,imgLowRes.shape[0],blockSize/8), -0.5, imgLowRes.shape[1]+0.5,linewidth=0.2,color='k')
plt.xlim(0,imgLowRes.shape[1])
plt.ylim(0,imgLowRes.shape[0])
plt.xticks([],[])
plt.yticks([],[])
saveFile=os.path.join(pltSaveDir,'Fig-1C1.png')
plt.savefig(saveFile,bbox_inches='tight',dpi=400)

#%% Individual plots below only 3 out of 4 image pairs are used
name='Fig1C1-'

rcRange=slice(500-int(blockSize/2),500+int(blockSize/2))
for i in range(4):
    plt.figure(figsize=(10,20))    
    fontSize=10
    plt.subplot(1,2,1)
    plt.imshow(nimgList[i][rcRange,rcRange,:])
    plt.xticks([],[])
    plt.yticks([],[])
    plt.subplot(1,2,2)
    plt.imshow(nimgList[i][rcRange,rcRange,:])
    plt.imshow(maskList[i][rcRange,rcRange],vmin=0,vmax=1,cmap=cmap_green,alpha=alpha1)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.tight_layout()
    figName=name+str(i)+'.png'
    saveFile=os.path.join(pltSaveDir,figName)
    plt.savefig(saveFile,bbox_inches='tight',dpi=400)
