#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:50:23 2022
This is for training only trains on AngioScores
Use Multi Batch MSE loss just like the mixed model

Several variations of model are used:
    
    ResNet Model  -  Is 224x224 input and Just ResNet18 as is ..with 1 class instead of 1000 class output
    This model is trained 3 different ways
    
    
    Variation-Base Case - Train ResNet18 as is  Train all layers [ modelType == 0]
    variation-1 is - ResNet18 with all layers frozen except only FC layer trained [ modelType == 1]
    variation-2 is -  ResNet18 where all layers are frozen except last convolution, and FC layer [modelType ==2]


"""
import os as os
import yaml
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(__file__))
print(parent_dir)
sys.path.insert(0, parent_dir)
                
import numpy as np
import pickle
from imgaug import augmenters as iaa
#import extendedImgaugAugmenters as exiaa

import pandas as pd

from tqdm import tqdm


import torch
from torch.utils.data import Dataset, DataLoader,Sampler
from torchvision import models 

import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
import Data_Generation.PatchGen as pg
import Data_Generation.extendedImgaugAugmenters as exiaa
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


#%% read Yaml Data

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
modelType=yaml_data['modelType']

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
    
    if fold <3:
        trainIdx=trainFoldsIdx[fold]
        testIdx=testFoldsIdx[fold]
        
    else:
        trainIdx=np.concatenate([trainFoldsIdx[0],testFoldsIdx[0]])


# now that we have the trainIdx, and testIdx find UTSW scores for these IDS
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


# %%


class ScoreImgDataSet(Dataset):
    def __init__(self,scorePatches,scores,scoreSampleNumbers,
                transform=None,preproc_fn= lambda x:np.float32(x)/255):

        self.scorePatches=scorePatches
        self.scores=scores
        self.nScores=len(scores)

        self.numberOfPatches=self.nScores
        
        self.sampleNumbers=scoreSampleNumbers
        self.uSamples=np.unique(self.sampleNumbers).tolist()

        self.transform=transform
        self.preproc=preproc_fn



    def __len__(self):
        return self.numberOfPatches

    def __getitem__(self,idx):
        
        assert idx>=0 and idx <self.numberOfPatches
        
        sample=int(self.sampleNumbers[idx])

        img=self.scorePatches[idx]
        score=self.scores[idx] 


        if self.transform is not None:            
            img=np.squeeze(self.transform(images=np.expand_dims(img,0)))
       
        img=self.preproc(img).transpose(2,0,1)
        
        sampleOh=np.zeros(len(self.uSamples))
        sampleOh[self.uSamples.index(sample)]=1

        return {'image':img,'score':score,'sample':sampleOh}
    
class MixedSampler(Sampler):
    
    def __init__(self,dataset,batchSizeScore,nSamplesPerScoreBatch):
        self.nBatchScore=int(np.floor(dataset.nScores/batchSizeScore))
      
        self.numBatches=self.nBatchScore
        self.nSamplesPerScoreBatch=nSamplesPerScoreBatch
        self.sampleNumbers=dataset.sampleNumbers

        self.batchSizeScore=batchSizeScore

        self.uniqueSampleNumbersScore=np.unique(self.sampleNumbers[self.sampleNumbers>=0])
        self.sampleIdxScore={}
        for sampleNumber in self.uniqueSampleNumbersScore:
              self.sampleIdxScore[sampleNumber]=np.where(self.sampleNumbers==sampleNumber)[0]
              
        self.pSampleScore=np.array([len(self.sampleIdxScore[sampleNumber]) 
                        for sampleNumber in self.uniqueSampleNumbersScore])/dataset.nScores


    def __iter__(self):


        self.indexList=[]

        for i in range(self.numBatches):
            
            patchesPerSample=np.ceil(self.batchSizeScore/self.nSamplesPerScoreBatch)
            patchToSampleNumber=np.uint(np.floor(np.arange(self.batchSizeScore)/patchesPerSample))
            sampleNumbersInBatch=np.random.choice(self.uniqueSampleNumbersScore,size=self.nSamplesPerScoreBatch,
                                          p=self.pSampleScore)
            sampleNumbers=sampleNumbersInBatch[patchToSampleNumber]
            self.indexList.append([np.random.choice(self.sampleIdxScore[s]) for s in sampleNumbers])
             
             
        return iter(self.indexList)
    
    def __len__(self):
        return self.numBatches  
    

# %%
batchSizeScore=32
nSamplesPerScoreBatch=4


np.random.seed(seedNum)

augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([
    exiaa.HEDadjust(
        hAlpha = (0.975, 1.025), eAlpha = (0.975, 1.025), rAlpha = (0.975, 1.025)),
    iaa.color.RemoveSaturation(mul=[0.0,0.5])]) ])

                                                                                                                                          
augmentations.seed_(seedNum)

trainDataset=ScoreImgDataSet(trainPatchesScore,trainScores,trainSampleNumbersScore,
                             transform=augmentations)
trainBatchSampler=MixedSampler(trainDataset,batchSizeScore,nSamplesPerScoreBatch)
trainLoader=DataLoader(trainDataset,batch_sampler=trainBatchSampler)


#%% 3 versions of ResNet18 models
# running 3 possibilities  : freeze every thing except the fc layer
#  and same as above + unfreeze last conv layer
# 3rd case is where no layers are frozen

# the first is where no layers are frozen
rNet=models.resnet18(pretrained=True)

num_features = rNet.fc.in_features
# change the layer to output only 1 class instead of 1000
rNet.fc=nn.Linear(num_features, 1)
# class created  in case we want to add more layers
class RNeTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.topNetwork=rNet
    def forward(self,x):
        
        out=torch.flatten(self.topNetwork(x))
        return out

angioModel=RNeTModel()
if modelType == 0:
    pass    # model already defined
elif modelType == 1:
# this is the scenario where all layers are frozen except fc layer
    for name, param in angioModel.named_parameters():
        print(name, param.numel())
        if 'fc' not in name :
            param.requires_grad = False
elif modelType == 2:
# freeze two layers both fc and layer4.1

    for name, param in angioModel.named_parameters():
        print(name, param.numel())
        if 'fc' in name or 'layer4.1' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
else:
    print('Invalid Model Type')

#%%
def count_parameters(model):                                                                                                                    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
param_count = count_parameters(angioModel)                                                                                                       
print('trainable params: ',param_count)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
angioModel= angioModel.to(device)


# %%
numEpochs=10


def CalcLossRNAMulti(predAngio,trueAngio,samplesOh,metrics):
    sampleWeights=torch.sum(samplesOh,axis=0,keepdims=True)
    nSamples=torch.sum(sampleWeights)
    sampleWeights[sampleWeights==0]=1
    samplesOh=samplesOh/sampleWeights
    
    trueAngioSample=torch.einsum('b,bs->s', trueAngio, samplesOh)
    predAngioSample=torch.einsum('b,bs->s', predAngio, samplesOh)

    angioLoss=torch.sum(torch.square(trueAngioSample-predAngioSample))/nSamples

    metrics['angioLoss'] += angioLoss.data.cpu().numpy() * trueAngio.size(0)

    loss=angioLoss
    return loss


#optimizer = optim.SGD(filter(lambda p: p.requires_grad, angioModel.parameters()), lr=1e-4,momentum=0.9)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, angioModel.parameters()), lr=1e-4)
#%%
for epoch in range(numEpochs):
    print('Epoch {}/{}'.format(epoch, numEpochs - 1))
    print('-' * 10)
    since = time.time()
    
    angioModel.train()  # Set model to training mode
    
    metrics = defaultdict(float)
    rnaSamples = 1


    with tqdm(trainLoader,unit='batch') as tepoch:
        for batch in tepoch:
            inputImg = batch['image'].to(device)   
            trueAngio = batch['score'].to(device).float()
            samplesOh=batch['sample'].to(device).float()
    
            # zero the parameter gradients
            optimizer.zero_grad()

                # forward
                # track history if only in train
            with torch.set_grad_enabled(True):
                predAngio= angioModel(inputImg)
                loss = CalcLossRNAMulti(predAngio,trueAngio,samplesOh, metrics)
                rnaSamples+=inputImg.shape[0] 
    
                loss.backward()
                optimizer.step()
 
                tepoch.set_postfix(angioLoss=metrics['angioLoss']/rnaSamples)

    
        modelSaveDir=modelDir
        modelSaveFile=os.path.join(modelSaveDir,'Encoder_MultiBatch_E'+str(epoch)+'.pt')
        angioModel.eval()
        torch.save(angioModel.state_dict(),modelSaveFile)
        angioModel.train()

