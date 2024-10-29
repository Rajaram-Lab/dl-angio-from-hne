#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 09:02:08 2023
 Mixed Model and ResNetUNets 


"""
import numpy as np
import torch

from torchvision import models

import torch.nn as nn


#%%
#base_model = models.resnet18(pretrained=False)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


def dfs_freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

class ResNetUNet(nn.Module):

    def __init__(self, n_class,freezeResnet=False, pretrained=False):
        super().__init__()

        self.base_model = models.resnet18(pretrained=False)
        
        
        self.base_model = models.resnet18(pretrained=pretrained)
        if freezeResnet:
            dfs_freeze(self.base_model)


        self.base_layers = list(self.base_model.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(64, 64, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)


    def crop_and_concat(self, upsampled, bypass):

        c = -(upsampled.size()[2]-bypass.size()[2]) // 2
        #bypass = F.pad(bypass, (-c, -c, -c, -c))
        bypass=bypass[:,:,c:(c+upsampled.size()[2]),c:(c+upsampled.size()[2])]
        return torch.cat((upsampled, bypass), dim=1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        
        neck = self.layer4_1x1(layer4)
        x = self.upsample(neck)
        layer3 = self.layer3_1x1(layer3)

        x = self.crop_and_concat(layer3, x)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x=self.crop_and_concat(layer2,x)

        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x=self.crop_and_concat(layer1,x)

        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x=self.crop_and_concat(layer0,x)

        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x=self.crop_and_concat(x_original,x)

        x = self.conv_original_size2(x)        

        out= self.conv_last(x)
        
        return out,neck

# %%
class MixedModelSimple(nn.Module):

    def __init__(self, UNet):
        super().__init__()
        
        # UNet Model: Image to Mask
        self.UNet= UNet
        
        # UNet Neck To Score Layers
        self.scoreConv1=convrelu(512,512,3,'same')        
        self.scoreConv2=convrelu(512,512,3,'same')     
        self.scoreConv3=convrelu(512,512,4,0)     
        self.scoreConv4=convrelu(512,512,4,0)     
        self.scoreConv5=convrelu(512,512,4,0)     
        self.scorePool=nn.MaxPool2d(2)
        self.scoreConv6=convrelu(512,1024,7,0)     
        self.scoreConv7=nn.Conv2d(1024,1,1)
        
        # Mask to MaskAngio
        self.mAngioAct=nn.Sigmoid()
        self.mAngioGAP=nn.AdaptiveAvgPool2d((1,1))
        self.mAngioN0=nn.Linear(1,128)    
        self.mAngioN1=nn.Linear(128,128)    
        self.mAngioFinal=nn.Linear(128,1)    
        
    
    def forward(self,imgIn):
        
        maskOut,neck=self.UNet(imgIn)
        
        x=self.scoreConv1(neck)
        x=self.scoreConv2(x)
        x=self.scoreConv3(x)
        x=self.scoreConv4(x)
        x=self.scoreConv6(x)
        scoreOut=torch.flatten((self.scoreConv7(x)))
        
        x=nn.functional.softmax(maskOut,dim=1)
        x=self.mAngioGAP(x)[:,1,:,:]
        x=nn.ReLU()(self.mAngioN0(x))
        x=nn.ReLU()(self.mAngioN1(x))
        maskAngioOut=torch.flatten(self.mAngioFinal(x))
        
        return maskOut,scoreOut,maskAngioOut




