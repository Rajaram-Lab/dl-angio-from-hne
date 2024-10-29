#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Evaluates UNet model, where model predicts the CD31 Mask for each patch
And percent positive CD31 marker is calculated for each mask
The final value is averaged over all the patches in a given sample
This code is specific to evaluating models on UTSEQ data
There are nuances to UTSEQ that is different from other cohorts where
multiple samples exist on a single slide and also there are potentially two slides for
each sample (top of the sample and flip side of sample)

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
huaPathwayFile=yaml_data['huaPathwayFile']
allSamplesFile=yaml_data['allSamplesFile']
patchSize=yaml_data['patchSize']
modelsToEval=yaml_data['modelsToEval']
# load the model

cd31UNetModel=MM.ResNetUNet(2)

# %%  
allHdf5List=glob.glob(os.path.join(patchDir,'*.hdf5'))
allSamplesInfo=pd.read_excel(allSamplesFile)
isGood=np.logical_not(pd.isnull(allSamplesInfo['Angiogenesis']))
allSamplesInfo=allSamplesInfo[isGood]

hdf5List=[]
hdf5ToSampleId={}
hdf5ToDirection={}
for i in range(allSamplesInfo.shape[0]):
    for slideDirection in ['Top','Flip']:
       svs=allSamplesInfo.iloc[i]['Image.'+slideDirection]
       batch=allSamplesInfo.iloc[i]['Batch']
       if isinstance(svs,str):
           
           svs=os.path.split(svs)[-1]
           hdf5=svs.replace('.svs','.hdf5')
           
           if os.path.exists(os.path.join(patchDir,hdf5)):
           #if hdf5 in hdf5PathDict:
               hdf5List.append(hdf5)
               if hdf5 in hdf5ToSampleId:
                   hdf5ToSampleId[hdf5].append(allSamplesInfo.iloc[i]['Sample.ID'])
               else:
                   hdf5ToSampleId[hdf5]=[allSamplesInfo.iloc[i]['Sample.ID']]
               hdf5ToDirection[hdf5]=slideDirection   
           else:
               print(hdf5,' not found')

hdf5List=np.unique(hdf5List).tolist()

huaPathwayScores=pd.read_csv(huaPathwayFile,sep='\t')

huaAngio=[huaPathwayScores.loc[pId]['Angiogenesis'] for pId in allSamplesInfo['Sample.ID']]
allSamplesInfo['HuaAngio']=huaAngio
allSamplesInfo=allSamplesInfo.set_index('Sample.ID')
#%%  


for indx,epoch in enumerate(yaml_data['epochs']):
    
    mixedModelFile=modelsToEval[indx]

    print('Working on model epoch # ', epoch, os.path.split(mixedModelFile)[-1])
    cd31UNetModel.load_state_dict(torch.load(mixedModelFile))
    cd31UNetModel=cd31UNetModel.to(device)
    cd31UNetModel.eval()
    

    trueScoresDict={}

    predPctPosDict={}
    patchPctPosPredDict={}
    
    for hdf5 in tqdm(hdf5ToSampleId):
        hdf5File=os.path.join(patchDir,hdf5)
        slideDirection=hdf5ToDirection[hdf5]
        patchData,patchClasses,classDict=pg.LoadPatchData(\
                                [hdf5File],returnSampleNumbers=False)
            
        
        slideData=allSamplesInfo.loc[hdf5ToSampleId[hdf5]]
        punchIdList=slideData['Punch.ID'].values.tolist()
        
        batch=slideData['Batch'].values[0]

        sourcePatches=patchData[0]   
        
        for anno in classDict:
            if anno in punchIdList:
                # find the row number corresponding to that annotation
                sampleData=slideData.iloc[punchIdList.index(anno)]
    
                isInAnno=patchClasses==classDict[anno]
                

                temp=patchData[0][isInAnno]
                sampleId=sampleData['Sample.ID.original']
                if sampleId not in predPctPosDict:

                    predPctPosDict[sampleId]={}
                    patchPctPosPredDict[sampleId]={}
                    
                inPatches=temp
                sampleDS=ImgDataSet(inPatches)
                        
                sampleDL=DataLoader(sampleDS,batch_size=16,shuffle=False)

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

                trueScoresDict[sampleId]=sampleData['HuaAngio']
                predPctPosDict[sampleId][slideDirection]=np.round(100.0*np.mean(pctCD31)/(patchSize*patchSize),4)
                patchPctPosPredDict[sampleId][slideDirection]=[np.round(100.0*cd31Frac/(patchSize*patchSize),4) for cd31Frac in pctCD31]
                   
            else:
                print(anno+' in '+hdf5+' not found. Skipping!')

    
    sampleNames=list(predPctPosDict)

    trueScores=[]

    topPctPosPred=[]
    flipPctPosPred=[]
    avgPctPosPred=[]
    
    for sampleId in sampleNames:
        if 'Top' in predPctPosDict[sampleId]:
            topPctPosPred.append(np.mean(predPctPosDict[sampleId]['Top']))
        else:
            topPctPosPred.append(np.NAN)
        if 'Flip' in predPctPosDict[sampleId]:
            flipPctPosPred.append(np.mean(predPctPosDict[sampleId]['Flip']))
        else:
            flipPctPosPred.append(np.NAN)

        avgPctPosPred.append(np.mean([np.mean(predPctPosDict[sampleId][slideDir]) for slideDir in ['Top','Flip'] 
                                if slideDir in predPctPosDict[sampleId]]))
        if sampleId in trueScoresDict:
            trueScores.append(trueScoresDict[sampleId])
        else:
            trueScores.append(np.NAN)    

    
    top_col_name2='Pred_Pct_Pos_Top'+'E_'+str(epoch)
    flip_col_name2='Pred_Pct_Pos_Flip'+'E_'+str(epoch)
    mean_col_name2='Pred_Pct_Pos_Mean'+'E_'+str(epoch)
    
    if not os.path.exists(saveFile):
        pilotDf=pd.DataFrame({'Sample':sampleNames,'RNA_Angio':trueScores})
            
    pilotDf[top_col_name2]=topPctPosPred
    pilotDf[flip_col_name2]=flipPctPosPred
    pilotDf[mean_col_name2]=avgPctPosPred

    pilotDf.to_csv(saveFile,index=False)
