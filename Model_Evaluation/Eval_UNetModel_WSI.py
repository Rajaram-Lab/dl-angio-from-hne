#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Evaluates UNet model, where model predicts the CD31 Mask for each patch
And percent positive CD31 marker is calculated for each mask
The final value is averaged over all the patches in a given sample(or slide)

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
import matplotlib.pyplot as plt

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
# sctter plot
def ScatterWithCorr(x,y,titlePrefix='',**kwargs):                                                                                                                                        
    plt.scatter(x,y,10,c='gray')                                                                                                                                                         
    pCorr,pVal=Corr(x,y,measure='pearson')                                                                                                                                               
    sCorr,pVal=Corr(x,y,measure='spearman')                                                                                                                                              
                                                                                                                                                                   
    plt.title(titlePrefix+'pCorr:'+str(round(pCorr,2))+',sCorr:'+str(round(sCorr,2)),                                                                                                    
              pad=-8,fontsize=10)

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
patchDir=yaml_data['patchDir']
saveFile=yaml_data['saveFile']
modelsToEval=yaml_data['modelsToEval']
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

    hneFiles=[os.path.split(name)[-1].replace('hdf5','svs') for name in hdfFiles]
    resDf=pd.DataFrame({'SVS':hneFiles})

# %%  initialize Model

cd31UNetModel=MM.ResNetUNet(2)

#%%
predPctCD31={}

for indx,epoch in enumerate(yaml_data['epochs']):
    
    modelFile=modelsToEval[indx]
    colName2='PctPos_F0'
 
    predPctCD31[colName2]=[]

    print('Working on model epoch # ', epoch, os.path.split(modelFile)[-1])
    cd31UNetModel.load_state_dict(torch.load(modelFile))
    cd31UNetModel=cd31UNetModel.to(device)
    cd31UNetModel.eval()
    
    for hneName in tqdm(hneFiles):
        if 'svs' in hneName:
            hdf5Name=hneName.replace('.svs','.hdf5')
        else:
            hdf5Name=hneName.replace('.ndpi','.hdf5')
        hdf=os.path.join(patchDir,hdf5Name)
        if os.path.exists(hdf):
            patchData,patchClasses,classDict=pg.LoadPatchData([hdf],returnSampleNumbers=False)
    
            sourcePatches1=patchData[0]
            sourcePatches=RemoveBgPatches(sourcePatches1)
              
            inPatches=sourcePatches
            sampleDS=ImgDataSet(inPatches)
            sampleDL=DataLoader(sampleDS,batch_size=32,shuffle=False)


            pctCD31=[]
            predMasks=[]
            with torch.inference_mode():
                for batch in sampleDL:
                    imgs=batch['image'].float().to(device)
                    if imgs.shape[0] >0 :
                        predMask,_=cd31UNetModel(imgs)
                        predMasks.append(predMask.cpu().detach().numpy())
                predMasksNp=np.concatenate(predMasks)
                for i in range(predMasksNp.shape[0]):
                    mask=predMasksNp[i].transpose(1,2,0)
                    pctCD31.append(np.sum(np.argmax(mask,axis=2)))
            
            predPctCD31[colName2].append(np.round(100.0*np.mean(pctCD31)/(patchSize*patchSize),4))
        else:
            predPctCD31[colName2].append(np.nan)
 

for columnName in predPctCD31:
    resDf[columnName] = predPctCD31[columnName]
#resDf.to_csv(saveFile,index=False)


#%%  Find correlation
colName=colName2
xField='Angioscore'
ScatterWithCorr(resDf[xField].values,resDf[colName].values,'PctPos Epoch 7')
