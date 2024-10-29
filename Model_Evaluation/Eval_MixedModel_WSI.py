#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
'''
Created on Sun Dec 11 10:23:06 2022
This file reads patches and evaluates Mixed Model output
Tabluates results for each slide (sample) as a mean of all 
Reads patches for each slide (patient) and predicts Mean scores

'''
import os as os
import yaml
import sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
print(parent_dir)
sys.path.insert(0, parent_dir)

import glob
import numpy as np
import argparse

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import scipy.stats as stats
import Model_Training.MixedModel as MM
import Data_Generation.PatchGen as pg

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# %% Convenience Functions

def Corr(x,y,isGood=None,measure='spearman'):                                                                                                            
    if isGood is None:                                                                                                                
        isGood=np.logical_and(np.isfinite(x),np.isfinite(y))  
    assert measure in ['spearman','pearson']
    if measure == 'spearman':                                                      
        corr,p=stats.spearmanr(x[isGood],y[isGood])  
    else:    
        corr,p=stats.pearsonr(x[isGood],y[isGood])                                                                               
    return corr,p

def RemoveBgPatches(inPatches,intThreshold=240,pctFgThreshold=0.4):
    res=np.any(inPatches<intThreshold,axis=-1)
    fracFg=res.sum(axis=tuple(range(1,3)))/np.prod(inPatches.shape[1:3])
    return inPatches[fracFg>pctFgThreshold]

def preprocess():
    #ToTensor() normalizes to 0 to 1 and changes dimensions
    # from (H x W x C) to (C x H x W)
    # and then weird normalization..
    transform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),])
    
    return transform

class ImgDataSet(Dataset):
    def __init__(self,patches,
                transform=None,preproc_fn= lambda x:np.float32(x)/255):

        self.patches=patches
        self.transform=transform
        self.transform1=preprocess()
        self.preproc=preproc_fn
    def __len__(self):
        return len(self.patches)

    def __getitem__(self,idx):

        img=self.patches[idx]

        if self.transform is not None:
            img=np.squeeze(self.transform(img))
        
        #img=self.transform1(img)
        img=self.preproc(img).transpose(2,0,1)

        return {'image':img}

# %% Load Params from Yaml


yamlFileName=args.yamlFile
   
assert os.path.isfile(yamlFileName), f"File '{yamlFileName}' does not exist"
                                                                  
with open(yamlFileName) as file:                                                                                      
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)                                                                    
print(yaml_data)

#%%
# 
modelDir=yaml_data['modelDir']
patchDir=yaml_data['patchDir']
saveFile=yaml_data['saveFile']
patchSize=yaml_data['patchSize']

#%%
if os.path.exists(saveFile):
    resDf=pd.read_csv(saveFile)
    hneFiles=resDf[yaml_data['starterColName']].to_list()
elif os.path.exists(yaml_data['starterFile']):
    resDf=pd.read_csv(yaml_data['starterFile'])
    hneFiles=resDf[yaml_data['starterColName']].to_list()
else:
    hdfFiles=glob.glob(os.path.join(yaml_data['patchDir'],'*.hdf5'))

    if yaml_data['starterColName'] == "ndpiFile":
        hneFiles=[os.path.split(name)[-1].replace('hdf5','ndpi') for name in hdfFiles]
        resDf=pd.DataFrame({'ndpiFile':hneFiles})
    else:
        hneFiles=[os.path.split(name)[-1].replace('hdf5','svs') for name in hdfFiles]
        resDf=pd.DataFrame({'SVS':hneFiles})



#%%
modelFiles=[os.path.split(f)[-1] for f in glob.glob(os.path.join(modelDir,'*.pt'))]
# model Files -order them by epoch number
modelEpochs=[name.split('_E')[1].split('.')[0] for name in modelFiles]
modelEpochInts=[int(val) for val in modelEpochs]
#%%
modelsToEval=[]
for epoch in yaml_data['epochs']:
    indx=modelEpochInts.index(epoch)
    modelName=modelFiles[indx]
    print(epoch, modelName)
    assert(os.path.exists(os.path.join(modelDir,modelName)))
    modelsToEval.append(os.path.join(modelDir,modelName))

# %%  Load Model

angioMixedModel=MM.MixedModelSimple(MM.ResNetUNet(2))

#%%
predAngioScores={}
predMaskAngioScores={}
predPctCD31={}


for indx,epoch in enumerate(yaml_data['epochs']):
    
    mixedModelFile=modelsToEval[indx]
    colName='Mixed_Angio_F0_E'+str(epoch)
    colName1='Mixed_MaskAngio_F0_E'+str(epoch)
    colName2='PctPos_F0_E'+str(epoch)
    predAngioScores[colName]=[]
    predMaskAngioScores[colName1]=[]
    predPctCD31[colName2]=[]
    print('Working on model epoch # ', epoch, os.path.split(mixedModelFile)[-1])
    angioMixedModel.load_state_dict(torch.load(mixedModelFile))
    angioMixedModel=angioMixedModel.to(device)
    angioMixedModel.eval()
    
    for hneName in tqdm(hneFiles):
        if 'svs' in hneName:
            hdf5Name=hneName.replace('.svs','.hdf5')
        else:
            hdf5Name=hneName.replace('.ndpi','.hdf5')
        hdf=os.path.join(patchDir,hdf5Name)
        if os.path.exists(hdf):
            patchData,patchClasses,classDict=pg.LoadPatchData([hdf],returnSampleNumbers=False)
    
            sourcePatches1=patchData[0]
            inPatches=RemoveBgPatches(sourcePatches1)
            sampleDS=ImgDataSet(inPatches)
            sampleDL=DataLoader(sampleDS,batch_size=32,shuffle=False)
            
            angioScores=[]
            maskAngioScores=[]
            pctCD31=[]
            predMasks=[]
            with torch.inference_mode():
                for batch in sampleDL:
                    imgs=batch['image'].float().to(device)
                    if imgs.shape[0] >0 :
                        predMask,predAngio,predMAngio=angioMixedModel(imgs)
                        angioScores+=predAngio.cpu().detach().numpy().tolist()
                        maskAngioScores+=predMAngio.cpu().detach().numpy().tolist()
                        predMasks.append(predMask.cpu().detach().numpy())
                predMasksNp=np.concatenate(predMasks)
                for i in range(predMasksNp.shape[0]):
                    mask=predMasksNp[i].transpose(1,2,0)
                    pctCD31.append(np.sum(np.argmax(mask,axis=2)))
            
            predAngioScores[colName].append(np.round(np.mean(angioScores),4))
            predMaskAngioScores[colName1].append(np.round(np.mean(maskAngioScores),4))
            predPctCD31[colName2].append(np.round(100.0*np.mean(pctCD31)/(patchSize*patchSize),4))
        else:
            predAngioScores[colName].append(np.nan)
            predMaskAngioScores[colName1].append(np.nan)
            predPctCD31[colName2].append(np.nan)
 
for columnName in predAngioScores:
    resDf[columnName]=predAngioScores[columnName]
for columnName in predMaskAngioScores:
    resDf[columnName]=predMaskAngioScores[columnName]
for columnName in predPctCD31:
    resDf[columnName] = predPctCD31[columnName]
resDf.to_csv(saveFile,index=False)
#%%
corrList1=[]
corrList2=[]
corrList3=[]
xField='TestScores'
#xField='Angioscore'

for epoch in yaml_data['epochs']:

    colName='Mixed_Angio_F0_E'+str(epoch)
    corr,pVal=Corr(resDf[xField].values,resDf[colName].values)
    colName1='Mixed_MaskAngio_F0_E'+str(epoch)
    corr1,pVal=Corr(resDf[xField].values,resDf[colName1].values)
    colName2='PctPos_F0_E'+str(epoch)
    corr2,pVal=Corr(resDf[xField].values,resDf[colName2].values)
    corrsum=corr+corr1+corr2
    print (epoch, np.round(corr,4), np.round(corr1,4), np.round(corr2,4),np.round(corrsum,4))
    corrList1.append(np.round(corr,3))
    corrList2.append(np.round(corr1,4))
    corrList3.append(np.round(corr2,4))
