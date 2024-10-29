#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:50:23 2022

 for training the Mixed Model

"""
import os as os
import yaml
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(__file__))
print(parent_dir)
sys.path.insert(0, parent_dir)

import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from imgaug import augmenters as iaa

from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import pandas as pd
#import seaborn as sns
from tqdm import tqdm
   

import torch
from torch.utils.data import Dataset, DataLoader,Sampler


import torch.nn as nn
import torch.optim as optim

from collections import defaultdict

import Model_Training.MixedModel as MM
import Model_Training.Dice as Dice
import Data_Generation.extendedImgaugAugmenters as exiaa

import Data_Generation.PatchGen as pg
import time

#%%

def parse_args():                                                                                                                                     
    """Parse input arguments.                                                                                                                         
    Parameters                                                                                                                                        
    -------------------                                                                                                                               
    No parameters.                                                                                                                                    
    Returns                                                                                                                                           
    -------------------                                                                                                                               
    args: argparser.Namespace class object                                                                                                            
        An argparse.Namespace class object contains experimental hyper-parameters.                                                                    
    """                                                                                                                                               
    parser = argparse.ArgumentParser(description='Automated job submission')                                                                          
                                                                                                                                                      
                                                                                                                                                      
    parser.add_argument('--yamlFile', dest='yamlFile',type=str)

    args = parser.parse_args()
    return args

args=parse_args()


#%%

yamlFileName=args.yamlFile
   
assert os.path.isfile(yamlFileName), f"File '{yamlFileName}' does not exist"
                                                                  
with open(yamlFileName) as file:                                                                                      
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)                                                                    
print(yaml_data)


modelDir=yaml_data['modelDir']
nEpoch=yaml_data['nEpoch']
seedNum=yaml_data['seedNum']
foldNum=yaml_data['foldNum']
patchDir=yaml_data['patchDir']
patchSize=yaml_data['patchSize']
maskPatchDir= yaml_data['maskPatchDir']
test_files = yaml_data['test_files']
cd31SaveDir = yaml_data['cd31SaveDir']
cd31File=os.path.join(cd31SaveDir,yaml_data['cd31File'])
# %% Setup RNA files: H&E patches + angio scores (from TCGA)

friendlyNameMappingFile=yaml_data['mappingFile']      
friendlyNameMapping=pd.read_csv(friendlyNameMappingFile)                                                                               
friendlyNames=[os.path.split(s)[-1].replace('.svs','.hdf5')                                                                            
               for s in friendlyNameMapping['Friendly Path'].values]                                                                   
tcgaIds=['-'.join(os.path.split(s)[-1].split('-')[:3])                                                                                 
               for s in friendlyNameMapping['Original Path'].values]                                                                   
tcgaLongIds=[('-'.join(os.path.split(s)[-1].split('-')[:4])[:-1])                                                                      
               for s in friendlyNameMapping['Original Path'].values]

foldsSaveFile=yaml_data['foldsSaveFile']
trainFoldsIdx,testFoldsIdx=pickle.load(open(foldsSaveFile,'rb'))

print(' Started working on Fold: ', foldNum)

for fold in range(3):

    if fold != foldNum:
        continue
    
    trainIdx=trainFoldsIdx[fold]
    testIdx=testFoldsIdx[fold]


#%% now that we have the trainIdx, and testIdx find UTSW scores for these IDS
pathwaysFile=yaml_data['pathwaysFile']
pathwayData=pd.read_csv(pathwaysFile,sep='\t')
rnaTcgaIds=[s[:-1] for s in pathwayData.index]
pathwayData['ids']=rnaTcgaIds
pathwayData=pathwayData.set_index('ids')


trainTcgaIdsLong=np.array(tcgaLongIds)[trainIdx]
trainSampleScores=pathwayData.loc[trainTcgaIdsLong]['Angiogenesis'].values              
assert len(trainTcgaIdsLong) ==len(trainSampleScores)

testTcgaIdsLong=np.array(tcgaLongIds)[testIdx]
testSampleScores=pathwayData.loc[testTcgaIdsLong]['Angiogenesis'].values              
assert len(testTcgaIdsLong) ==len(testSampleScores)

trainHdf5List=[os.path.join(patchDir,f) 
               for f in np.array(friendlyNames)[trainIdx]]
testHdf5List=[os.path.join(patchDir,f) 
               for f in np.array(friendlyNames)[testIdx]]

assert len(trainHdf5List)==len(trainSampleScores)
assert len(testHdf5List)==len(testSampleScores)
assert np.all([os.path.exists(hdf5) for hdf5 in trainHdf5List])
assert np.all([os.path.exists(hdf5) for hdf5 in testHdf5List])

#%% Load Score Patches
trainPatchesScore,_,_,trainSampleNumbersScore=pg.LoadPatchData(trainHdf5List,returnSampleNumbers=True)
trainPatchesScore=trainPatchesScore[0]
trainSampleNumbersScore=np.uint32(trainSampleNumbersScore)
trainScores=trainSampleScores[trainSampleNumbersScore]
print('Train Patches Shape:', trainPatchesScore.shape)

# %% Load Mask Patches

testList=[os.path.join(maskPatchDir,name) for name in test_files]
allFiles=glob.glob(os.path.join(maskPatchDir,'*.pkl'))
fileList=[name for name in allFiles if name not in testList]


trainPatchesMask=[]
trainMasks=[]
for patchFile in tqdm(fileList):
    with open(patchFile,'rb') as pFileHandle:
        hneImg,ihcImg,ihcMask,cDict=pickle.load(pFileHandle)
        trainPatchesMask.append(hneImg)
        trainMasks.append(ihcMask)

trainPatchesMask=np.concatenate(trainPatchesMask)
trainMasks=np.concatenate(trainMasks)
classDict=cDict
# %%


class MixedImgDataSet(Dataset):
    def __init__(self,scorePatches,scores,scoreSampleNumbers,
                 maskPatches,masks,
                transform=None,preproc_fn= lambda x:np.float32(x)/255):

        self.scorePatches=scorePatches
        self.scores=scores
        self.nScores=len(scores)
        
        self.maskPatches=maskPatches
        self.masks=masks
        self.nMasks=masks.shape[0]

        self.numberOfPatches=self.nMasks+self.nScores

        
        self.sampleNumbers=-1*np.ones(self.numberOfPatches,dtype=np.int32)
        self.sampleNumbers[0:len(scores)]=scoreSampleNumbers
        self.uSamples=np.unique(self.sampleNumbers).tolist()
        
        
        self.isScores=np.ones(self.numberOfPatches)
        self.isScores[len(scores):]=0
 
        assert masks.shape[1]==self.scorePatches.shape[1]   
        self.patchSize=masks.shape[1]
 
        self.transform=transform
        self.preproc=preproc_fn



    def __len__(self):
        return self.numberOfPatches

    def __getitem__(self,idx):
        
        assert idx>=0 and idx <self.numberOfPatches
        
        sample=int(self.sampleNumbers[idx])
        isScore=self.isScores[idx]
        
        if idx<self.nScores: # Get Scores
            img=self.scorePatches[idx]
            score=self.scores[idx] 
            mask=np.zeros((self.patchSize,self.patchSize))
            
        else: #Get Masks
            idx1=idx-self.nScores
            img=self.maskPatches[idx1]
            score=-1
            mask=self.masks[idx1]


        if self.transform is not None:
            if isScore:
                img=np.squeeze(self.transform(images=np.expand_dims(img,0)))
            else:
                segmap=SegmentationMapsOnImage(np.int32(mask),
                                               shape=(self.patchSize,self.patchSize))
                img,mask=self.transform(images=np.expand_dims(img,0),
                         segmentation_maps=segmap)
                img=np.squeeze(img)
                mask=mask.get_arr()


        
        img=self.preproc(img).transpose(2,0,1)
        
        sampleOh=np.zeros(len(self.uSamples))
        sampleOh[self.uSamples.index(sample)]=1

        return {'image':img,'score':score,'mask':mask,'sample':sampleOh,'isScore':isScore}
    
class MixedSampler(Sampler):
    
    def __init__(self,dataset,batchSizeScore,batchSizeMask,nSamplesPerScoreBatch):
        self.nBatchScore=int(np.floor(dataset.nScores/batchSizeScore))
        self.nBatchMask=int(np.floor(dataset.nMasks/batchSizeMask))
        
        self.numBatches=self.nBatchMask+self.nBatchScore

        self.nSamplesPerScoreBatch=nSamplesPerScoreBatch
        self.sampleNumbers=dataset.sampleNumbers

        self.batchSizeScore=batchSizeScore
        self.batchSizeMask=batchSizeMask
        
        
        
        self.uniqueSampleNumbersScore=np.unique(self.sampleNumbers[self.sampleNumbers>=0])
        self.sampleIdxScore={}
        for sampleNumber in self.uniqueSampleNumbersScore:
              self.sampleIdxScore[sampleNumber]=np.where(self.sampleNumbers==sampleNumber)[0]
              
        self.pSampleScore=np.array([len(self.sampleIdxScore[sampleNumber]) 
                        for sampleNumber in self.uniqueSampleNumbersScore])/dataset.nScores
        
        self.maskIdxList=np.arange(dataset.nMasks)+dataset.nScores

    def __iter__(self):


        batchTypeList=np.concatenate((np.ones(self.nBatchScore),np.zeros(self.nBatchMask)))
        np.random.shuffle(batchTypeList)
        np.random.shuffle(self.maskIdxList)


        maskBatchCounter=0
        self.indexList=[]
               
        for isBatchScore in batchTypeList:
            if isBatchScore:
                patchesPerSample=np.ceil(self.batchSizeScore/self.nSamplesPerScoreBatch)
                patchToSampleNumber=np.uint(np.floor(np.arange(self.batchSizeScore)/patchesPerSample))
                sampleNumbersInBatch=np.random.choice(self.uniqueSampleNumbersScore,size=self.nSamplesPerScoreBatch,
                                              p=self.pSampleScore)
                sampleNumbers=sampleNumbersInBatch[patchToSampleNumber]
                self.indexList.append([np.random.choice(self.sampleIdxScore[s]) for s in sampleNumbers])
            else:
                start=maskBatchCounter*self.batchSizeMask
                self.indexList.append(self.maskIdxList[start:(start+self.batchSizeMask)])
                maskBatchCounter+=1                
             
        return iter(self.indexList)
    
    def __len__(self):
        return self.numBatches  
    

# %%
batchSizeScore=32
nSamplesPerScoreBatch=4
batchSizeMask=4

np.random.seed(seedNum)

augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([
    exiaa.HEDadjust(
        hAlpha = (0.975, 1.025), eAlpha = (0.975, 1.025), rAlpha = (0.975, 1.025)),
    iaa.color.RemoveSaturation(mul=[0.0,0.5])]) ])

                                                                                                                                          
augmentations.seed_(seedNum)

trainDataset=MixedImgDataSet(trainPatchesScore,trainScores,trainSampleNumbersScore,
                             trainPatchesMask,trainMasks,
                             transform=augmentations)
trainBatchSampler=MixedSampler(trainDataset,batchSizeScore,batchSizeMask,nSamplesPerScoreBatch)
trainLoader=DataLoader(trainDataset,batch_sampler=trainBatchSampler)


#%%
# initialize the model

cd31Unet = MM.ResNetUNet(2)
cd31Unet.load_state_dict(torch.load(cd31File))
angioMixedModel=MM.MixedModelSimple(cd31Unet)

# %%
makePlots=False

if makePlots:
    tempBatch=next(iter(trainLoader))
    predMask,predScore,preMAngio=angioMixedModel(tempBatch['image'])
    for n in range(batchSizeMask):
        mask=tempBatch['mask'][n].detach().cpu().numpy()
        img=tempBatch['image'][n].detach().cpu().numpy().transpose(1,2,0)
        pMask=np.argmax(predMask[n].detach().cpu().numpy(),axis=0)
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.xticks([],[])
        plt.yticks([],[])

        plt.subplot(1,3,2)    
        if tempBatch['isScore'][0]==0:

            plt.imshow(img)
            plt.imshow(mask,vmin=0,vmax=1,cmap=colors.ListedColormap(['w','g']),alpha=0.5)
        else:
            plt.axis('square')
            plt.title('NO CD31 GT')
        plt.xticks([],[])
        plt.yticks([],[])
        
        plt.subplot(1,3,3)
        plt.imshow(img)
        plt.imshow(pMask,vmin=0,vmax=1,cmap=colors.ListedColormap(['w','g']),alpha=0.5)
        plt.xticks([],[])
        plt.yticks([],[])
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
angioMixedModel= angioMixedModel.to(device)
    
    
# %% Calculate Weights
numberOfClasses=2
classWeights=[]
for c in range(numberOfClasses):
    classWeights.append(np.sum(trainMasks==c))
classWeights=np.array(classWeights)
classWeights=torch.from_numpy(np.max(classWeights)/classWeights).type(torch.float32).to(device)

# %%
numEpochs=10
beta=0.9
cce=nn.CrossEntropyLoss(weight=classWeights)
dice=Dice.DiceLoss()

def CalcLossCD31(predMask, predMaskAngio,predAngio,trueMask, metrics):
    lossCCE = cce(predMask, trueMask)
    lossDice=dice(predMask, trueMask)
    maskLoss=((1-beta)*lossCCE)+(beta*lossDice)    
    scoreConsistencyLoss=torch.mean(torch.square(predMaskAngio-predAngio))
    loss=maskLoss+scoreConsistencyLoss
    metrics['maskLoss'] += maskLoss.data.cpu().numpy() * predMask.size(0)
    metrics['consLoss'] += scoreConsistencyLoss.data.cpu().numpy() * predMask.size(0)
    return loss

def CalcLossRNA(predMaskAngio,predAngio,trueAngio, metrics):
    angioLoss=torch.square(torch.mean(trueAngio)-torch.mean(predAngio))
    maskAngioLoss=torch.square(torch.mean(predMaskAngio)-torch.mean(trueAngio))
    metrics['angioLoss'] += angioLoss.data.cpu().numpy() * trueAngio.size(0)
    metrics['mAngioLoss'] += maskAngioLoss.data.cpu().numpy() * trueAngio.size(0)
    loss=angioLoss+maskAngioLoss
    return loss

def CalcLossRNAMulti(predMaskAngio,predAngio,trueAngio,samplesOh,metrics):
    sampleWeights=torch.sum(samplesOh,axis=0,keepdims=True)
    nSamples=torch.sum(sampleWeights)
    sampleWeights[sampleWeights==0]=1
    samplesOh=samplesOh/sampleWeights
    
    trueAngioSample=torch.einsum('b,bs->s', trueAngio, samplesOh)
    predAngioSample=torch.einsum('b,bs->s', predAngio, samplesOh)
    predMaskAngioSample=torch.einsum('b,bs->s', predMaskAngio, samplesOh)
    
    angioLoss=torch.sum(torch.square(trueAngioSample-predAngioSample))/nSamples
    maskAngioLoss=torch.sum(torch.square(trueAngioSample-predMaskAngioSample))/nSamples
    
    metrics['angioLoss'] += angioLoss.data.cpu().numpy() * trueAngio.size(0)
    metrics['mAngioLoss'] += maskAngioLoss.data.cpu().numpy() * trueAngio.size(0)
    loss=angioLoss+maskAngioLoss
    return loss



optimizer = optim.SGD(filter(lambda p: p.requires_grad, angioMixedModel.parameters()), lr=1e-4,momentum=0.9)
    

for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        print('-' * 10)
        
        since = time.time()
            
        angioMixedModel.train()  # Set model to training mode

        metrics = defaultdict(float)
        rnaSamples = 1
        cd31Samples=1


        with tqdm(trainLoader,unit='batch') as tepoch:
            for batch in tepoch:
                inputImg = batch['image'].to(device)
                trueMask = batch['mask'].type(torch.LongTensor).to(device)    
                trueAngio = batch['score'].to(device).float()
                samplesOh=batch['sample'].to(device).float()
                isRNA=batch['isScore'][0]
                # zero the parameter gradients
                optimizer.zero_grad()

                        # forward
                        # track history if only in train
                with torch.set_grad_enabled(True):
                    predMask,predAngio,predMAngio= angioMixedModel(inputImg)
                    if isRNA:
                        
                        #loss = CalcLossRNA(predMAngio,predAngio,trueAngio, metrics)
                        loss = CalcLossRNAMulti(predMAngio,predAngio,trueAngio,samplesOh, metrics)
                        rnaSamples+=inputImg.shape[0]
                    else:
                        
                        loss=CalcLossCD31(predMask, predMAngio,predAngio,trueMask, metrics)
                        cd31Samples+=inputImg.shape[0]

                    loss.backward()
                    optimizer.step()


                    tepoch.set_postfix(mLoss=metrics['maskLoss']/cd31Samples,
                                       consLoss=metrics['consLoss']/cd31Samples,
                                       angioLoss=metrics['angioLoss']/rnaSamples,
                                       mAngioLoss=metrics['mAngioLoss']/rnaSamples)
            
            modelSaveDir=modelDir
            modelSaveFile=os.path.join(modelSaveDir,'MixedSimple_UNetResnet_E'+str(epoch)+'.pt')
            angioMixedModel.eval()
            torch.save(angioMixedModel.state_dict(),modelSaveFile)
            angioMixedModel.train()

