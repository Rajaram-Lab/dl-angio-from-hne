#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:20:26 2023
Supplementary Figure 3
This generates CD31 masks from held out data using the ResNet UNet 
from the final Mixed model

"""
# %%
import os as os
import yaml

import numpy as np

from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages

import torch

from torch.utils.data import Dataset, DataLoader


import Model_Training.MixedModel as MM

# %%
codeDir=os.path.dirname(__file__)
homeDir = os.path.dirname(codeDir)

figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)


globalParamsFile=os.path.join(homeDir,'Global_Params.yml')


globalParams=yaml.safe_load(open(globalParamsFile,'r'))

saveDir=globalParams['Plotting']['saveDir']
predName=globalParams['Plotting']['methodNames']['H&E Prediction'] 
pointColor=globalParams['Plotting']['methodColors'][predName]  

#%%
def getImages_Masks(testFile, angioMixedModel):
    
    with open(testFile,'rb') as pFileHandle:
        hneImg,ihcImg,ihcMask,cDict=pickle.load(pFileHandle)
        
    patchSize=416
    batchSize=8
    testDataset=ImgMaskDataSet(hneImg,ihcMask,patchSize)
    testLoader=DataLoader(testDataset,batch_size=batchSize,shuffle=False,num_workers=16)
    predMasks=[]
    for batch_indx,batch in enumerate(testLoader):
        imgs=batch['image'].float().to(device)
        #evaluate in eval model
        if imgs.shape[0] >1 :
            with torch.inference_mode():
                predMask,predAngio,predMAngio=angioMixedModel(imgs)
                predMasks.append(predMask.cpu().detach().numpy())
    predMasksNp=np.concatenate(predMasks)
    
    return hneImg,ihcImg,ihcMask,predMasksNp
class ImgMaskDataSet(Dataset):
    def __init__(self,imgList,maskList,patchSize,
                transform=None,preproc_fn= lambda x:np.float32(x)/255):

        self.imgList=imgList
        self.numberOfPatches=len(imgList)
        self.maskList = maskList        
        self.patchSize=patchSize
        self.transform=transform
        self.preproc=preproc_fn

    def __len__(self):
        return self.numberOfPatches

    def __getitem__(self,idx):
        
        img=self.imgList[idx]
        mask=self.maskList[idx]
       
        if self.transform is not None:
            segmap=SegmentationMapsOnImage(np.int32(mask),shape=(self.patchSize,self.patchSize))
            img,mask=self.transform(images=np.expand_dims(img,0),
                                     segmentation_maps=segmap)
            img=np.squeeze(img)
            mask=mask.get_arr()

        img=self.preproc(img).transpose(2,0,1)
        return {'image':img,'mask':mask}
    
def slide_prf(hneImg,ihcImg,gtMasks,predMasksNp):

    tot_common=0
    tot_falseN=0
    tot_falseP=0

    for imgnum in range(gtMasks.shape[0]):
        gtMask=gtMasks[imgnum]
        modelMask=predMasksNp[imgnum].transpose(1,2,0)
        predMask=np.argmax(modelMask,axis=2)

        score=predMask+2*gtMask

        common=np.sum(score==3)                                                                                                                    
        falseP=np.sum(score==1)                                                                                                                    
        falseN=np.sum(score==2) 
        tot_common=tot_common+common                                                                                                               
        tot_falseN=tot_falseN+falseN                                                                                                               
        tot_falseP=tot_falseP+falseP
        
    gt=tot_common+tot_falseN
    # reported positive                                                                                                                       
    rep_pos=tot_common+tot_falseP 
                                                                                                             
    if rep_pos == 0:                                                                                                                               
        precision=0                                                                                                                                
    else:                                                                                                                                          
        precision=round(float(tot_common)/float(rep_pos), 2)                                                                                       
    recall = round(float(tot_common)/float(gt),2)                                                                                                  
                                                                                                                                                   
    f1 = 2*precision*recall/(precision+recall)                                                                                                     
    f11=round(f1,2)  

    return precision,recall,f11

def print_pdf_2(pdf_file,ihcimgs,ihcmasks,hneImgs,masks1):
    
    imgPerPage=8
    colsPerImg=4
    imgCounter=1
    fontSize=14
    rangemax=len(ihcimgs)
    mSize=416
    pctAct=[]
    pct12=[]

    ssimVals=[]
    imgNums=[]
    pctSkelAct=[]
    pctSkel1=[]

    with PdfPages(pdf_file) as export_pdf:
        for imgnum in range(0,rangemax):
            imgNums.append(imgnum)    
            gtmask1=ihcmasks[imgnum]
            
            model1Mask=masks1[imgnum].transpose(1,2,0)
            pMask1=np.argmax(model1Mask,axis=2)


            if imgnum % imgPerPage == 0:
                fig=plt.figure(figsize=(30,60))
                
            plt.subplot(imgPerPage,colsPerImg,(imgCounter-1)*colsPerImg+1)
            plt.imshow(hneImgs[imgnum])
            plt.title('H&E '+str(imgnum),fontsize=fontSize)
            plt.xticks([],[])                                                                                                                                                     
            plt.yticks([],[])

            plt.subplot(imgPerPage,colsPerImg,(imgCounter-1)*colsPerImg+2)
            plt.imshow(ihcimgs[imgnum])
            plt.title('CD31 IHC',fontsize=fontSize)
            plt.xticks([],[])                                                                                                                                                     
            plt.yticks([],[])

            plt.subplot(imgPerPage,colsPerImg,(imgCounter-1)*colsPerImg+3)
            plt.imshow((hneImgs[imgnum]))
            plt.imshow(gtmask1,vmin=0,vmax=1,cmap=colors.ListedColormap(['w','g']),alpha=0.7)
            frac=round(100.0*np.sum(gtmask1)/(mSize*mSize),3)
            plt.title('GT %pos  '+str(frac),fontsize=fontSize)
            pctAct.append(frac)            
            plt.xticks([],[])                                                                                                                                                     
            plt.yticks([],[])
            
            plt.subplot(imgPerPage,colsPerImg,(imgCounter-1)*colsPerImg+4)
            plt.imshow((hneImgs[imgnum]))
            plt.imshow(pMask1,vmin=0,vmax=1,cmap=colors.ListedColormap(['w','g']),alpha=0.7)
            frac=round(100.0*np.sum(pMask1)/(mSize*mSize),3)
            plt.title('Pred %pos '+str(frac),fontsize=fontSize)
            pct12.append(frac)
            plt.xticks([],[])                                                                                                                                                     
            plt.yticks([],[])
                        
            imgCounter=imgCounter+1
            #if(imgCounter > imgPerPage):
            if (imgnum+1) % imgPerPage ==0 or (imgnum+1)== rangemax:
                plt.savefig(export_pdf,format='pdf',bbox_inches='tight')
                imgCounter=1
                plt.close()

        plt.figure(figsize=(16,8))
        
        plt.subplot(1,2,1)
        plt.hist(pctAct,density=True,bins=20)
        plt.xlabel('GT '+' mean: '+str(round(np.mean(pctAct),3)),fontsize=18)


        plt.subplot(1,2,2)
        plt.hist(pct12,density=True,bins=20)
        #plt.xlabel('12File UnetModel stain Fraction '+' mean: '+str(round(np.mean(pct12),2)),fontsize=18)
        plt.xlabel('Pred  '+' mean: '+str(round(np.mean(pct12),2)),fontsize=18)

        plt.savefig(export_pdf,format='pdf',bbox_inches='tight')
        plt.close()


    print('Area fraction mean values: GT, Unet pred',round(np.mean(pctAct),3),
          round(np.mean(pct12),3))
    return 
#%% Load test files
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_files=figuresParams['FigS3TestFiles']
patchDir=figuresParams['FigS3patchDir']



#%% Load the model
modelFile=figuresParams['FigS3ModelFile']

angioMixedModel=MM.MixedModelSimple(MM.ResNetUNet(2))
angioMixedModel.load_state_dict(torch.load(modelFile))                                                                             
angioMixedModel=angioMixedModel.to(device)                                                                                              
angioMixedModel.eval()

#%%  
imgIndx=figuresParams['FigS3imgIndx']

hneImgsPlt=[]
ihcImgsPlt=[]
ihcMasksPlt=[]
predMasksPlt=[]

for testFile in test_files:
    testFileFp=os.path.join(patchDir, testFile)
    hneImgs,ihcImgs,ihcMasks,predMasksNps=getImages_Masks(testFileFp,angioMixedModel)
    indxList=imgIndx[testFile]
    for i in indxList:
        hneImgsPlt.append(hneImgs[i])
        ihcImgsPlt.append(ihcImgs[i])
        ihcMasksPlt.append(ihcMasks[i])
        model1Mask=predMasksNps[i].transpose(1,2,0)
        pMask1=np.argmax(model1Mask,axis=2)
        predMasksPlt.append(pMask1)


#%%  Set Colormap 
pltNum=1
alpha1=0.6 
lineColor1=[0.0,1.0,0.0] # green 
colorsList=[(1,1,1,0),(lineColor1[0],lineColor1[1],lineColor1[2],alpha1)]                                                                                                                                                                          
cmap_green=colors.ListedColormap(colorsList)
plt.figure(figsize=(32,24))
for indx in range(3):

    plt.subplot(3,4,pltNum)
    plt.imshow(hneImgsPlt[indx])
    plt.xticks([],[])                                                                                                                                                     
    plt.yticks([],[])
    plt.subplot(3,4,pltNum+1)
    plt.imshow(ihcImgsPlt[indx])
    plt.xticks([],[])                                                                                                                                                     
    plt.yticks([],[])
    plt.subplot(3,4,pltNum+2)
    plt.imshow(hneImgsPlt[indx])
    plt.imshow(ihcMasksPlt[indx],vmin=0,vmax=1,cmap=cmap_green,alpha=alpha1)
    plt.xticks([],[])                                                                                                                                                     
    plt.yticks([],[])
    plt.subplot(3,4,pltNum+3)
    plt.imshow(hneImgsPlt[indx])
    plt.imshow(predMasksPlt[indx],vmin=0,vmax=1,cmap=cmap_green,alpha=alpha1)
    plt.xticks([],[])                                                                                                                                                     
    plt.yticks([],[])
    pltNum=pltNum+4
    plt.tight_layout()   

saveFile=os.path.join(saveDir,'FigS3.png')
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)
