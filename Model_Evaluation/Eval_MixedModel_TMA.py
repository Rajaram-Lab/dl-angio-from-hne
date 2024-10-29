#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:59:44 2022
This file reads all patch data and calculates angio scores for each TMA punch
Using the model.
TMA sets refering as J or J-145 are TMA1 dataset
TMA sets refering as V or Vitaly are TMA2 dataset

"""
import numpy as np
import yaml
import os
import sys
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(__file__))
print(parent_dir)
sys.path.insert(0, parent_dir)

from tqdm import tqdm
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import Model_Training.MixedModel as MM
import Data_Generation.PatchGen as pg
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
#%% Load Params from Yaml
yamlFileName=args.yamlFile
   
assert os.path.isfile(yamlFileName), f"File '{yamlFileName}' does not exist"
                                                                  
with open(yamlFileName) as file:                                                                                      
    yamlData = yaml.load(file, Loader=yaml.FullLoader)                                                                    
print(yamlData)

#%% helper methods
def preprocess():

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
        
        img=self.preproc(img).transpose(2,0,1)

        return {'image':img}

# this calculates angio scores given the model 
# inputs are 
#       hdf5List :  full path of all patches for a given set
#       yamlData :  input data structure
#       angioModel:  model for predicting angio scores
#       saveFile:  csv file where the results are saved
#       colName;  Sno_id or Case_Identifier
#  returns nothing

def Calc_AngioScores(hdf5List,yamlData,angioModel,saveFile,colName):
    
    patchData,patchClasses,classDict,patchSampleNumbers=pg.LoadPatchData(hdf5List,returnSampleNumbers=True)    
    patchData=patchData[0]
    classToPunchName = {v: k for k, v in classDict.items()}
    
    angioScoreDict={}
    maskAngioDict={}
    pctPosDict={}    
    for hdf5Num in tqdm(range(len(hdf5List))):
        isInHdf5=patchSampleNumbers==hdf5Num
        classesInHdf5=patchClasses[isInHdf5]
        uClasses,classNumbers=np.unique(classesInHdf5,return_inverse=True)
        
        inPatches=patchData[isInHdf5]       
        sampleDS=ImgDataSet(inPatches)
        sampleDL=DataLoader(sampleDS,batch_size=16,shuffle=False)
        angioScores=[]
        maskAngioScores=[]
        pctCD31=[]
        predMasks=[]
        #if True:
        with torch.inference_mode():
            for batch in sampleDL:
                imgs=batch['image'].float().to(device)
                if imgs.shape[0] >0 :
                    predMask,predAngio,predMAngio=angioModel(imgs)
                    angioScores+=predAngio.cpu().detach().numpy().tolist()
                    maskAngioScores+=predMAngio.cpu().detach().numpy().tolist()
                    predMasks.append(predMask.cpu().detach().numpy())
            predMasksNp=np.concatenate(predMasks)
            for i in range(predMasksNp.shape[0]):
                mask=predMasksNp[i].transpose(1,2,0)
                pctCD31.append(np.sum(np.argmax(mask,axis=2)))
        
        angioScoresNp=np.array(angioScores)
        maskAngioScoresNp=np.array(maskAngioScores)
        pctCD31Np=np.array(pctCD31)
        
        for c in range(len(uClasses)):
            angioScoreDict[classToPunchName[uClasses[c]]]=np.mean(angioScoresNp[classNumbers==c])
            maskAngioDict[classToPunchName[uClasses[c]]]=np.mean(maskAngioScoresNp[classNumbers==c])
            pctPosDict[classToPunchName[uClasses[c]]]=np.mean(pctCD31Np[classNumbers==c])

    svsnames=[]
    punchnames=[]
    case_ids=[]
    angio_Scores=[]
    maskAngio_Scores=[]
    pctPos_Scores=[]
    
    # new files names in Vitaly has few underscores in them
    for key in angioScoreDict:
        vals=key.split('_')
        if len(vals) == 3:
            svsnames.append(vals[0])
            punchnames.append(vals[1])
            case_ids.append(int(vals[2]))
        elif len(vals)==4:
            svsnames.append(vals[0]+'_'+vals[1])
            punchnames.append(vals[2])
            case_ids.append(int(vals[3]))
        elif len(vals)==5:
            svsnames.append(vals[0]+'_'+vals[1]+'_'+vals[2])
            punchnames.append(vals[3])
            case_ids.append(int(vals[4]))
        angio_Scores.append(angioScoreDict[key])
        maskAngio_Scores.append(maskAngioDict[key])
        pctPos_Scores.append(pctPosDict[key])
    angioDf=pd.DataFrame({colName: case_ids, 'SVS': svsnames,'Punch':punchnames,'AngioScore': angio_Scores,
                          'maskAngioScore':maskAngio_Scores,'pctPos': pctPos_Scores})
    angioDf.to_csv(saveFile,index=False)
    
    return

#%%
# load the angio Model.. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixedModelFile=yamlData['modelFile']
angioModel=MM.MixedModelSimple(MM.ResNetUNet(2))
angioModel.load_state_dict(torch.load(mixedModelFile))
angioModel=angioModel.to(device)
angioModel.eval()
#%%
# process and save J-145 results
patchSaveDir=yamlData['patchSaveDir']
slideNums=yamlData['slidesJ']
hdf5Files=[name+'.hdf5' for name in slideNums]
hdf5List=[os.path.join(patchSaveDir, name) for name in hdf5Files]
#%%
# calculate angio scores fro J145 and save results
saveFile=yamlData['scoreSaveFileJ']
colName='Sno ID'
Calc_AngioScores(hdf5List,yamlData,angioModel,saveFile,colName)
#%%
# repeat for second TMA set
slideNums=yamlData['slidesV']
hdf5Files=[name+'.hdf5' for name in slideNums]
hdf5List=[os.path.join(patchSaveDir, name) for name in hdf5Files]
#%%
# calculate angio scores fro second TMA set and save results
saveFile=yamlData['scoreSaveFileV']
colName='Case_identifier'
Calc_AngioScores(hdf5List,yamlData,angioModel,saveFile,colName)

