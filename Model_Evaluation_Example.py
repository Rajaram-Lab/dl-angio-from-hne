#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:48:06 2024

This reads sample images, and loads the  mixed model 
and evaluate the images and  displays results 


"""

import os as os

import sys

import yaml
import argparse

parent_dir = os.path.dirname(os.path.dirname(__file__))

current_dir = os.path.dirname(__file__)

print(parent_dir, current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import Model_Training.MixedModel as MM

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
# %% Load Params from Yaml


yamlFileName=args.yamlFile
   
assert os.path.isfile(yamlFileName), f"File '{yamlFileName}' does not exist"
                                                                  
with open(yamlFileName) as file:                                                                                      
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)                                                                    
print(yaml_data)

#%%
# 

imageDir=os.path.join(current_dir,yaml_data['imageDir'])

mixedModelFile=yaml_data['modelFile']
print(current_dir)
print(imageDir)
# %% Convenience Functions

class ImgDataSet(Dataset):
    def __init__(self,patches,
                transform=None,preproc_fn= lambda x:np.float32(x)/255):

        self.patches=patches
        self.transform=transform
        self.preproc=preproc_fn
    def __len__(self):
        return len(self.patches)

    def __getitem__(self,idx):

        img=self.patches[idx]

        if self.transform is not None:
            img=np.squeeze(self.transform(img))
        
        img=self.preproc(img).transpose(2,0,1)

        return {'image':img}

def center_crop(img, crop_size):
    h, w, c = img.shape
    crop_h, crop_w = crop_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h+crop_h, start_w:start_w+crop_w, :]

#%% read sample images


file_list = [f for f in os.listdir(imageDir) if f.endswith('.jpg') or f.endswith('.jpeg')]
img_list = []

# Read each image into a NumPy array and add it to the list
for file in file_list:
    file_path = os.path.join(imageDir, file)
    img = imageio.v2.imread(file_path)
    img_416 = center_crop(img, (416, 416))
    img_list.append(img_416)

# Convert the list to a NumPy array
img_array = np.array(img_list)

patchSize=img_list[0].shape[0]
#%%
sampleDS=ImgDataSet(img_array)
sampleDL=DataLoader(sampleDS,batch_size=4,shuffle=False)

#%%

# %%  Load Model
angioMixedModel=MM.MixedModelSimple(MM.ResNetUNet(2))
angioMixedModel.load_state_dict(torch.load(mixedModelFile))
angioMixedModel=angioMixedModel.to(device)

#%%
angioScores=[]
maskAngioScores=[]
pctCD31=[]

with torch.inference_mode():
    for batch in sampleDL:
        imgs=batch['image'].float().to(device)
        if imgs.shape[0] >0 :
            predMask,predAngio,predMAngio=angioMixedModel(imgs)
            angioScores+=predAngio.cpu().detach().numpy().tolist()
            maskAngioScores+=predMAngio.cpu().detach().numpy().tolist()
            predClasses=torch.argmax(predMask,dim=1)
            masks=predClasses.detach().cpu().numpy()
            pctCD31+=(torch.sum(predClasses,dim=[1,2])/(patchSize*patchSize)).cpu().detach().numpy().tolist()

angioScores = np.round(angioScores, 2)
maskAngioScores = np.round(maskAngioScores, 2)
pctCD31 = np.round(pctCD31, 2)
#%%  display images and scores
# angioScores are from the angio arm
# maskAngioScores are from the mask arm
# percent positive is from the mask area pct of marker
fontSize=14
alpha1=0.6 
lineColor1=[0.0,1.0,0.0] # green 
colorsList=[(1,1,1,0),(lineColor1[0],lineColor1[1],lineColor1[2],alpha1)]                                                                                                                                                                          
cmap_green=colors.ListedColormap(colorsList)

for i in range(4):
    plt.figure(figsize=(10,20))    
    fontSize=10
    plt.subplot(1,2,1)
    plt.title('Scores: '+'AngioArm: '+str(angioScores[i])+' MaskArm: '+str(maskAngioScores[i]))
    plt.imshow(img_array[i])
    plt.xticks([],[])
    plt.yticks([],[])
    plt.subplot(1,2,2)
    plt.imshow(img_array[i])
    plt.imshow(masks[i],vmin=0,vmax=1,cmap=cmap_green,alpha=alpha1)
    plt.title('Predicted Mask, '+'Fraction :'+str(pctCD31[i]))
    plt.xticks([],[])
    plt.yticks([],[])
    plt.tight_layout()
    plt.show()
#%%
