#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 05:50:21 2022

This code trains the segmentation model using CD31 masks

Several models are trained by varying alpha

Alpha is varied between 0.5 - 0.9 (which is a multiplier to Dice)

The model at alpha 0.9 was used as the final model


"""
# %%
import os as os
import yaml
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(__file__))
print(parent_dir)
sys.path.insert(0, parent_dir)

import glob
import numpy as np
from tqdm import tqdm
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import pickle

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time


from imgaug import augmenters as iaa



from collections import defaultdict
import Model_Training.MixedModel as MM
import Model_Training.Dice as Dice
import Data_Generation.extendedImgaugAugmenters as exiaa

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


modelSaveDir=yaml_data['modelDir']
patchDir=yaml_data['patchDir']
test_files=yaml_data['test_files']
patchSize=yaml_data['patchSize']
num_epochs= yaml_data['nEpoch']
#%% read Patches and Masks, exclude test list

testList=[os.path.join(patchDir,name) for name in test_files]
allFiles=glob.glob(os.path.join(patchDir,'*.pkl'))
fileList=[name for name in allFiles if name not in testList]


trainHneImgs=[]
trainTargetMasks=[]
for patchFile in tqdm(fileList):
    with open(patchFile,'rb') as pFileHandle:
        hneImg,ihcImg,ihcMask,cDict=pickle.load(pFileHandle)
        trainHneImgs.append(hneImg)
        trainTargetMasks.append(ihcMask)

trainHneImgs=np.concatenate(trainHneImgs)
trainTargetMasks=np.concatenate(trainTargetMasks)
classDict=cDict
print('Number of Patches: ', trainHneImgs.shape[0])
# %%
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


# %%

numberOfClasses=len(classDict)
batchSize=16

augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([
    exiaa.HEDadjust(
        hAlpha = (0.975, 1.025), eAlpha = (0.975, 1.025), rAlpha = (0.975, 1.025)),
    iaa.color.RemoveSaturation(mul=[0.0,0.5])]) ])

trainDataset=ImgMaskDataSet(trainHneImgs,trainTargetMasks,patchSize,
                    transform=augmentations)

trainLoader=DataLoader(trainDataset,batch_size=batchSize,shuffle=True,num_workers=16)


# %% Get Class Weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classWeights=[]
for c in range(numberOfClasses):
    classWeights.append(np.sum(trainTargetMasks==c))
classWeights=np.array(classWeights)
classWeights=torch.from_numpy(np.max(classWeights)/classWeights).type(torch.float32).to(device)

# %%
# loop over alpha which is a multiplier to Dice.. see totalLoss definition below
for alpha in np.linspace(0.5,0.9,5):
    model = MM.ResNetUNet(numberOfClasses)

    model = model.to(device)
        
    cce=nn.CrossEntropyLoss(weight=classWeights)
    dice=Dice.DiceLoss()
    def calc_loss(pred, target, metrics,alpha):
        lossCCE = cce(pred, target)
        lossDice=dice(pred,target)
        totalLoss=((1-alpha)*lossCCE)+(alpha*lossDice)
        metrics['loss'] += totalLoss.data.cpu().numpy() * target.size(0)
        metrics['lossCCE'] += lossCCE.data.cpu().numpy() * target.size(0)
        metrics['lossDice'] += lossDice.data.cpu().numpy() * target.size(0)
        return totalLoss
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)      
    
    for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
            since = time.time()
    
            model.train()  # Set model to training mode    
    
            metrics = defaultdict(float)
            epoch_samples = 0    
    
            with tqdm(trainLoader,unit='batch') as tepoch:
                for batch in tepoch:
                    inputs = batch['image'].to(device)
                    labels = batch['mask'].type(torch.LongTensor).to(device)    
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    with torch.set_grad_enabled(True):
                        outputs,neck = model(inputs)
                        loss = calc_loss(outputs, labels, metrics,alpha=alpha)
    
                        predClasses = outputs.argmax(dim=1, keepdim=True).squeeze()                
                        accuracy=100*(labels==predClasses).sum().item()/labels.numel()

                        metrics['accuracy']+=accuracy*labels.shape[0]
                        loss.backward()
                        optimizer.step()
    
                            # statistics
                        epoch_samples += inputs.size(0)
    
    
                        tepoch.set_postfix(loss=metrics['loss']/epoch_samples,
                                           dice=metrics['lossDice']/epoch_samples,
                                           cce=metrics['lossCCE']/epoch_samples,
                             accuracy=metrics['accuracy']/epoch_samples)
    
    
                modelSaveFile=os.path.join(modelSaveDir,'CD31_ResnetUNet_DiceCCE_A'+str(alpha)+'_E'+str(epoch)+'.pt')
                model.eval()
                torch.save(model.state_dict(),modelSaveFile)
                model.train()
                    
