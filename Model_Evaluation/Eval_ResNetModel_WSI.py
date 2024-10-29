#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Code for evaluating Various ResNet18 models

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
from torchvision import transforms, models

import torch.nn as nn

from tqdm import tqdm
import scipy.stats as stats
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
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# %% Load Params from Yaml
yamlFileName=args.yamlFile
   
assert os.path.isfile(yamlFileName), f"File '{yamlFileName}' does not exist"
                                                                  
with open(yamlFileName) as file:                                                                                      
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)                                                                    
print(yaml_data)

#%%
# Skip over all this code for Pilot evaluation as well as IMM b1b2 combined evaluation
modelDir=yaml_data['modelDir']
patchDir=yaml_data['patchDir']
saveFile=yaml_data['saveFile']


#%%
if os.path.exists(saveFile):
    resDf=pd.read_csv(saveFile)
    hneFiles=resDf[yaml_data['starterColName']].to_list()
elif os.path.exists(yaml_data['starterFile']):
    resDf=pd.read_csv(yaml_data['starterFile'])
    hneFiles=resDf[yaml_data['starterColName']].to_list()
else:
    hdfFiles=glob.glob(os.path.join(yaml_data['patchDir'],'*.hdf5'))

    hneFiles=[os.path.split(name)[-1].replace('hdf5','svs') for name in hdfFiles]
    resDf=pd.DataFrame({'SVS':hneFiles})



#%%
modelFiles=[os.path.split(f)[-1] for f in glob.glob(os.path.join(modelDir,'Encoder_MultiBatch_E*.pt'))]
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


#%% 3 versions of ResNet18 models for evaluation
# Load one of 3 possibilities  : freeze every thing except the fc layer
#  and same as above + unfreeze last conv layer
# 3rd case is where no layers are frozen

# For the purpose of evaluation we don't need to freeze any layers
# the code is shown for the sake of completeness

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
'''
# this is the scenario where all layers are frozen except fc layer
for name, param in angioModel.named_parameters():
    print(name, param.numel())
    if 'fc' not in name :
        param.requires_grad = False
'''
# freeze two layers both fc and layer4.1
'''
for name, param in angioModel.named_parameters():
    print(name, param.numel())
    if 'fc' in name or 'layer4.1' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
'''

#%%
predAngioScores={}

for indx,epoch in enumerate(yaml_data['epochs']):
    
    modelFile=modelsToEval[indx]
    colName='Angio_F0_E'+str(epoch)

    predAngioScores[colName]=[]

    print('Working on model epoch # ', epoch, os.path.split(modelFile)[-1])
    angioModel.load_state_dict(torch.load(modelFile))
    angioModel=angioModel.to(device)
    angioModel.eval()
    
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

            with torch.inference_mode():
                for batch in sampleDL:
                    imgs=batch['image'].float().to(device)
                    if imgs.shape[0] >0 :
                        predAngio=angioModel(imgs)
                        angioScores+=predAngio.cpu().detach().numpy().tolist()
            
            predAngioScores[colName].append(np.round(np.mean(angioScores),4))

        else:
            predAngioScores[colName].append(np.nan)

 
for columnName in predAngioScores:
    resDf[columnName]=predAngioScores[columnName]

resDf.to_csv(saveFile,index=False)
#%%
corrList1=[]
xField='TestScores'
#xField='Angioscore'
for epoch in yaml_data['epochs']:

    colName='Angio_F0_E'+str(epoch)
    corr,pVal=Corr(resDf[xField].values,resDf[colName].values)
    print (epoch, np.round(corr,4))
    corrList1.append(np.round(corr,3))

