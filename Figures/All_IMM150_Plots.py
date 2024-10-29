#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 07:35:11 2023
 This file generates Figures related to IMM150 data and some other data
 Fig. 2C, 2D, 4B, 4C, 4D,  and Supplementary Figures: S.6, S.8A, S.8B, S.12
 and S.13, S.14A, 14B, 14C

"""


survivalThresholdMethod='IMM150_Median' #'TCGA'
assert survivalThresholdMethod in ['TCGA','IMM150_Median']


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

import sys
import pandas as pd
import seaborn as sns


codeDir=os.path.dirname(__file__) 
homeDir = os.path.dirname(codeDir)


import scipy.stats as stats
from sklearn import metrics
from decimal import Decimal
import yaml

import matplotlib.ticker as ticker
from lifelines import KaplanMeierFitter,CoxPHFitter
from lifelines.utils import concordance_index

#%% Util Functions
def Corr(x,y,isGood=None,measure='spearman'):                                                                                                            
    if isGood is None:                                                                                                                
        isGood=np.logical_and(np.isfinite(x),np.isfinite(y))  
    assert measure in ['spearman','pearson']
    if measure == 'spearman':                                                      
        corr,p=stats.spearmanr(x[isGood],y[isGood])  
    else:    
        corr,p=stats.pearsonr(x[isGood],y[isGood])                                                                               
    return corr,p     

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

def plot(x,y,xlable, ylable, titleS,xlabelColor,yLabelColor):                                                                                                                 
    markerSize=10                                                                                                                                     
    fontSize=24                                                                                                                                       
    isGood=(np.logical_and(np.isfinite(x),np.isfinite(y)))                                                                                            
    #plt.figure(figsize=(10,10))                                                                                                                       
    plt.plot(x[isGood],y[isGood],'ok',markersize=markerSize)                                                                                          
    plt.xlabel(xlable,fontsize=fontSize,color=xlabelColor)                                                                                                              
    plt.ylabel(ylable,fontsize=fontSize,color=yLabelColor)                                                                                                              
    corr,p=Corr(x,y)                                                                                                    
    #plt.title(titleS+'  Corr:'+str(round(corr,3))+',p-val:'+('%2E' % Decimal(p)),fontsize=20)
    l=plt.legend(title='Corr:'+str(round(corr,3))+'\np-val:'+('%.2E' % Decimal(p)),
                 fontsize=24)
    plt.setp(l.get_title(),fontsize=fontSize)
# %% Load up Various Params

figuresYamlFile=os.path.join(codeDir,'Figures.yaml')
with open(figuresYamlFile, "r") as f:
    figuresParams = yaml.safe_load(f)
#%%
statAnnotionDir=figuresParams['statAnnotationDir']

globalParamsFile=os.path.join(homeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))
saveDir=globalParams['Plotting']['saveDir'] 
methodColors=globalParams['Plotting']['methodColors']
predName=globalParams['Plotting']['methodNames']['H&E Prediction']
armRenamingDict=globalParams['Plotting']['IMM150']['armNames']


sys.path.insert(0, statAnnotionDir)
from statannotations.Annotator import Annotator
#%% Load Up Saved IMM150 Results

saveFileCsv=figuresParams['imm150_File']
allRes=pd.read_csv(saveFileCsv)
metricList=['Angio', 'maskAngio', 'pctPos']
allRes['CD31']=[np.float32(s) if is_number(s) else np.NAN for s in allRes['%_CD31']]
allRes=allRes.rename({'TrueAngio':'RNA_Angio'},axis=1)

exclude_list=globalParams['Data']['IMM150']['failedQcFiles']
duplicate_list= globalParams['Data']['IMM150']['duplicateFiles']
removeList=exclude_list+duplicate_list
genentechRes=allRes[~allRes.FILENAME.isin(removeList)]
genentechRes=genentechRes.reset_index()


#%% Load Up Saved TCGA and UTSeq Results
tcgaResFile=figuresParams['TCGA_CV_File']
tcgaRes=pd.read_csv(tcgaResFile).set_index('SVS')

utSeqResFile=figuresParams['utseq_res_file']
utSeqRes=pd.read_csv(utSeqResFile).set_index('Sample')

#%% Fig 2: Scatter Plots comparing H&E DL Angio to RNA/CD31 
# all plots were printed with dpi=300  Fig.2C, Fig. 2D
#plt.savefig('filename.png', dpi=300)
# Fig 2c: IMM150 - RNA vs H&E DL Angio
plt.figure(figsize=(10,10))
plot(genentechRes['RNA_Angio'],genentechRes['pctPos'],'RNA Angio Score', \
     predName,'IMM150 Data',
     xlabelColor=methodColors['RNA Angio'],yLabelColor=methodColors[predName])
#saveFile=os.path.join(saveDir,'Fig2c.png')
#plt.savefig(saveFile,bbox_inches='tight')    

# Fig 2d: IMM150 - CD31 vs H&E DL Angio
plt.figure(figsize=(10,10))
plot(genentechRes['RNA_Angio'],genentechRes['CD31'],'RNA Angio Score', \
     'CD31','IMM150 Data',
     xlabelColor=methodColors['RNA Angio'],yLabelColor=methodColors['CD31'])
#saveFile=os.path.join(saveDir,'Fig2d.png')
#plt.savefig(saveFile,bbox_inches='tight')


# %% Supplementary Figure 8 A & B: Batch Effects across Cohorts based on RNA and H&E DL Angio

utSeqResG=utSeqRes.copy()

genentechResG=genentechRes[['RNA_Angio','pctPos']].copy() 
genentechRes['FUHRMAN']=genentechRes['FUHRMAN'].fillna('.')
genentechResG['Grade']=[np.NAN if '.' in g else np.float32(g[-1]) for g in genentechRes['FUHRMAN']]

genentechResG['Cohort']='IMM150'

tcgaResG=tcgaRes.copy()
combinedResG=pd.concat([genentechResG,utSeqResG,tcgaResG])
combinedResG.rename(columns={'pctPos': predName}, inplace=True)
combinedResG.rename(columns={'RNA_Angio': 'RNA Angio Score'}, inplace=True)
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.swarmplot(data=combinedResG,x='Grade',y='RNA Angio Score',hue='Cohort',dodge=True,
              order=[1.0,2.0,3.0,4.0])
plt.ylabel('RNA Angio Score',color=methodColors['RNA Angio'])
plt.title('RNA Angio Score')
plt.subplot(1,2,2)
sns.swarmplot(data=combinedResG,x='Grade',y=predName,hue='Cohort',dodge=True,
              order=[1.0,2.0,3.0,4.0])
plt.ylabel(predName,color=methodColors[predName])
plt.title('H&E prediction')



#%% Get Cutoffs (currently set at median) for Low/High Angio Call 

fieldCutoffs={}
fieldCutoffs['RNA_Angio']=genentechRes['RNA_Angio'].median()
if survivalThresholdMethod=='TCGA':
    fieldCutoffs['pctPos']=5.66
else:
    fieldCutoffs['pctPos']=genentechRes['pctPos'].median()

fieldCutoffs['CD31']=genentechRes['CD31'].median()



#%% Fig 4B (plus hazard ration calculations for S11):

fieldsToAnalyze=['RNA_Angio','pctPos','CD31']
fieldNameSimpleDict={'RNA_Angio':'RNA Angio',
                    'pctPos': predName,
                    'CD31':'CD31'}
labelfontSize=12
fontSize=18

armList=list(armRenamingDict)
armsToPlot=['SUNITINIB']

readOutList=['Method','Arm','N_high','N_low','HR','HR_95Low','HR_95High','pVal','CI']
hazardDfCombined={r:[]for r in readOutList}


for armCounter,armName in enumerate(armList):
    
    armRes=genentechRes[genentechRes['ACTARM']==armName].copy()
    
    plt.figure(figsize=(7,7))
    for field in fieldsToAnalyze:
        fieldNameSimple=fieldNameSimpleDict[field]

        
        
        isGood=np.logical_and(np.logical_not(pd.isnull(armRes[field].values)),
                              np.logical_not(pd.isnull(armRes['PFSINV'].values)))
        fieldThresholds={field:lambda x: x> fieldCutoffs[field]}
        
        
        survDf=armRes[isGood].copy()
        survDf['PFSINV']=survDf['PFSINV']/30.4  # convert days to months
        isHigh=fieldThresholds[field](survDf[field].values)
        
        

        
        # Do Cox Proportional Hazards Calculation
        coxDf=pd.DataFrame({'Duration':survDf['PFSINV'],
                                 'isCensored':survDf['PFSINV.event'],
                                 'status':isHigh})
        cph=CoxPHFitter()
        cph.fit(coxDf,duration_col='Duration',event_col='isCensored')
        hR=cph.summary.loc['status']['exp(coef)']
        pVal=cph.summary.loc['status']['p']
        ci=concordance_index(survDf['PFSINV'],survDf[field].values,survDf['PFSINV.event'])
        
        # Record Results
        hazardDfCombined['Method'].append(fieldNameSimple)
        hazardDfCombined['HR'].append(hR)
        hazardDfCombined['HR_95Low'].append(cph.summary.loc['status']['exp(coef) lower 95%'])
        hazardDfCombined['HR_95High'].append(cph.summary.loc['status']['exp(coef) upper 95%'])
        
        hazardDfCombined['pVal'].append(pVal)
        hazardDfCombined['N_high'].append(np.sum(isHigh))
        hazardDfCombined['N_low'].append(np.sum(~isHigh))
        hazardDfCombined['Arm'].append(armRenamingDict[armName])
        hazardDfCombined['CI'].append(ci)
        
        if armName in armsToPlot:
            # Make Kaplan Meier Plot
            kmfHigh = KaplanMeierFitter().fit(survDf[isHigh]['PFSINV'],
                                              survDf[isHigh]['PFSINV.event'], label=fieldNameSimple+' High')
            kmfLow = KaplanMeierFitter().fit(survDf[~isHigh]['PFSINV'],
                                             survDf[~isHigh]['PFSINV.event'], label=fieldNameSimple+' Low')
            
            
            ax=kmfHigh.plot(ci_show=False,color=methodColors[fieldNameSimple],linestyle='-',
                            linewidth=2,show_censors=True)
            ax=kmfLow.plot(ci_show=False,color=methodColors[fieldNameSimple],linestyle='--',
                           linewidth=2,show_censors=True)
            
            plt.xlabel('Time (Months)', fontsize=fontSize)
            plt.ylabel('Progression-free Survival Probability',fontsize=fontSize)
            plt.legend(fontsize=labelfontSize)
        

    
hazardDfCombined=pd.DataFrame(hazardDfCombined)
pd.set_option('display.max_columns', None)
print(hazardDfCombined)
plt.show()



# %% Figure S12: Compare Hazard Ratio Across Arms and Methods
# one thing to note is plot.errorbar adds the error to HR values while                                                                         
# the confidence[95%] are exact values..need to subtract HR to get the errorbar  
capsize=3
plt.figure(figsize=(8,5))
sns.set(font_scale=1.15)
sns.set_style("white")
ax=sns.pointplot(data=hazardDfCombined,y='Arm',x='HR',hue='Method',
              dodge=0.35,join=False,palette=methodColors)
# Find the x,y coordinates for each point
x_coords = []
y_coords = []
errors=[]
barColors=[]
for point_pair in ax.collections:
    
    for x, y in point_pair.get_offsets():
        x_coords.append(x)
        y_coords.append(y)
        label=point_pair.get_label()
        idx=hazardDfCombined['HR'].values.tolist().index(x)
        assert hazardDfCombined['Method'].values[idx]==label
        err1=hazardDfCombined['HR'].values[idx]-hazardDfCombined['HR_95Low'].values[idx]
        err2=hazardDfCombined['HR_95High'].values[idx]-hazardDfCombined['HR'].values[idx]
        errors.append([err1,err2])
        #errors.append([hazardDfCombined['HR_95Low'].values[idx],
        #               hazardDfCombined['HR_95High'].values[idx]])
        barColors.append(methodColors[label])
errors=np.transpose(np.array(errors))
# Calculate the type of error to plot as the error bars
# Make sure the order is the same as the points were looped over
for i in range(len(x_coords)):
    ax.errorbar(x_coords[i], y_coords[i], xerr=errors[:,[i]],
        ecolor=barColors[i],linestyle='None',capsize=capsize)
plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
plt.xscale('log')
#plt.legend(loc='center left')
ax.get_xaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(x, ',')))
plt.xticks([0.1,0.25,0.5,1,2])

plt.ylabel('')
plt.show()


# %% Objective Response Rate Bar Plot (Supplementary Fig. 14)


armColors=globalParams['Plotting']['IMM150']['armColors']

barWidth=0.75 # thickness of each bar
catWidth=1 #spacing between Low/High bars in each arm
groupWidth=2.5 #spacing between arms
#groupLabelYPos=-0.075 # yPosition of the 'High N=xxx' label
#armLabelYPos=-0.115 #yPosition of the arm label e.g. 'Sunitinib'
groupLabelYPos=-0.080 # yPosition of the 'High N=xxx' label
armLabelYPos=-0.120 #yPosition of the arm label e.g. 'Sunitinib'
for field in fieldsToAnalyze: # Loop over readout e.g., RNA_Angio or Mask_Angio

    plt.figure()
   
    isGood=np.logical_and(np.logical_not(pd.isnull(genentechRes[field].values)),
                          np.logical_not(pd.isnull(genentechRes['BESRSPI'].values)))
    
    survDf=genentechRes[isGood]

    for armCounter,arm in enumerate(armRenamingDict): # Loop over arm i.e. drug treatment
        armNameSimple=armRenamingDict[arm]
        armDf=survDf[survDf['ACTARM']==arm]

        for highCounter,isHigh in enumerate([True,False]): # Loop over High/Low level of readout

            isInGroup=(armDf[field]>fieldCutoffs[field])==isHigh
            groupDf=armDf[isInGroup]
            nGroup=groupDf.shape[0]
            cumFrac=0
            x=groupWidth*armCounter+(highCounter*catWidth)

            for responseCounter,response in enumerate(['CR','PR']): # Loop response level

                nResp=np.sum(groupDf['BESRSPI']==response)
                frac=nResp/nGroup
                plt.bar(x,frac,bottom=cumFrac,width=barWidth,
                        color=np.array(armColors[armNameSimple][responseCounter])/255)
                if frac>0:                        
                    plt.text(x,cumFrac+(frac/2),
                        str(np.round(frac*100))+'%',color='w',
                        ha='center',va='center',fontsize=10)
                cumFrac+=frac

            groupLabel=['High','Low'][highCounter]+'\nn='+str(nGroup)
            plt.text(x,groupLabelYPos,groupLabel,ha='center')    

        plt.text(groupWidth*armCounter+(highCounter*catWidth/2),
                armLabelYPos,armNameSimple,ha='center',
                color=np.array(armColors[armNameSimple][1])/255)            

    plt.title(fieldNameSimpleDict[field])
    plt.ylabel('ORR')
    plt.xticks([],[])
    plt.show()
    
    
# %% Fig 4C showing Fraction of Sunitinib responders in Low/High Angio from 3 assays


armColors=globalParams['Plotting']['IMM150']['armColors']

barWidth=0.75 # thickness of each bar
catWidth=1 #spacing between Low/High bars in each arm
groupWidth=2.5 #spacing between arms
groupLabelYPos=-0.170 # yPosition of the 'High N=xxx' label
armLabelYPos=-0.120 #yPosition of the arm label e.g. 'Sunitinib'
plt.figure()
for fieldCounter,field in enumerate(fieldsToAnalyze): # Loop over readout e.g., RNA_Angio or Mask_Angio


    isGood=np.logical_and(np.logical_not(pd.isnull(genentechRes[field].values)),
                          np.logical_not(pd.isnull(genentechRes['BESRSPI'].values)))
    survDf=genentechRes[isGood]
    armCounter=0
    arm='SUNITINIB'

    armNameSimple=armRenamingDict[arm]
    armDf=survDf[survDf['ACTARM']==arm]

    for highCounter,isHigh in enumerate([True,False]): # Loop over High/Low level of readout

        isInGroup=(armDf[field]>fieldCutoffs[field])==isHigh
        groupDf=armDf[isInGroup]
        nGroup=groupDf.shape[0]
        cumFrac=0
        x=groupWidth*fieldCounter+(highCounter*catWidth)

        nResp=np.sum(groupDf['BESRSPI']=='CR')+np.sum(groupDf['BESRSPI']=='PR')
        frac=nResp/nGroup
        if isHigh:
            hatch=None
        else:
            hatch='////'
        plt.bar(x,frac,bottom=0,width=barWidth,
                color=methodColors[fieldNameSimpleDict[field]],hatch=hatch)

        

        groupLabel=['High','Low'][highCounter]+'\nn='+str(nGroup)
        plt.text(x,groupLabelYPos,groupLabel,ha='center')    


    plt.ylabel('Fraction of Responders')
    plt.xticks([0.5,3,5.5],['RNA','H&E','CD31'],fontsize=14)
plt.show()    

# %% Supplementary Fig. 13: Compare Angio Score for Different Response Levels across methods
# 
sunitinibRes=genentechRes[genentechRes['ACTARM']=='SUNITINIB'].copy()
sunitinibRes=sunitinibRes.rename(columns={'BESRSPI':'Response_Level'})
sunitinibRes=sunitinibRes[~pd.isnull(sunitinibRes['Response_Level'])]

responderLabels=['CR','PR']
isResponder=np.array([s in responderLabels for s in sunitinibRes['Response_Level']])

plt.figure(figsize=(12,5))

for fieldCounter,field in enumerate(fieldsToAnalyze):
    plt.subplot(1,3,fieldCounter+1)    
    ax=sns.swarmplot(data=sunitinibRes,x='Response_Level',y=field,
                  order=['CR','PR','SD','PD'],color=methodColors[fieldNameSimpleDict[field]])
    plt.ylabel(fieldNameSimpleDict[field])
    pairs=[('PR','SD'),('PR','PD'),('SD','PD')]
    annotator = Annotator(ax, pairs, data=sunitinibRes, x='Response_Level',y=field,
                          order=['CR','PR','SD','PD'])
    annotator.configure(test='Mann-Whitney', show_test_name=False, loc='outside',
                        text_format='simple')
    annotator.apply_and_annotate()
    
 
plt.tight_layout()
plt.show()
#%%  Figure 4D: ROC for predicting responders for the 3 methods  
aucs=[]

strs=[]
plt.figure(figsize=(5,5))

for fieldCounter,field in enumerate(fieldsToAnalyze):
    fieldNameSimple=fieldNameSimpleDict[field]
    isGood=np.isfinite(sunitinibRes[field].values)

    fpr, tpr, thresholds = metrics.roc_curve(isResponder[isGood], 
                                             sunitinibRes[field].values[isGood])
    auc=metrics.roc_auc_score(isResponder[isGood], sunitinibRes[field].values[isGood])
    aucs.append(np.round(auc,2))
    strs.append(fieldNameSimple+ ' ='+str(np.round(auc,2)))
    plt.plot(fpr,tpr,linewidth=1.5,color=methodColors[fieldNameSimple])
    #plt.plot([0,1],[0,1],'--',color='gray')  
    plt.axis('square')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.title('Response to Treatment')
plt.legend(strs,title='AUC',loc='lower right')
plt.plot([0,1],[0,1],'--',linewidth=1,color='gray')
plt.show()



#%%  Supplementary Figure 6


GenFile=figuresParams['imm150_all_data']
genDf=pd.read_csv(GenFile)
#%%
subj1=genDf[genDf['SUBJECT']==21553]
subj2=genDf[genDf['SUBJECT']==320046]
subj3=genDf[genDf['SUBJECT']==234741]
subj4=genDf[genDf['SUBJECT']==806992]
subj5=genDf[genDf['SUBJECT']==958651]
#%%
duplicate_list= globalParams['Data']['IMM150']['duplicateFiles']
exclude_list=globalParams['Data']['IMM150']['failedQcFiles']
removeList=exclude_list+duplicate_list+['SLIDEe80bba52aa_HE-2021-11-09.ndpi']
genRes=genDf[~genDf.FILENAME.isin(removeList)]

#%% MAP currenct cullection Types into a new Column called Sample Type
Collection_Map={"NEEDLE CORE":"BIOPSY","PUNCH BIOPSY":"BIOPSY",
                "NOT PROVIDED":"UNKNOWN","OTHER":"UNKNOWN",
                ".":"UNKNOWN","LEFT NEPHRECTONY":"RESECTION",
                "RADICAL NEPHRECTOMY":"RESECTION",
                "TOTAL NEPHRECTOMY":"RESECTION",
                "EXCISION":"RESECTION","NEPHRECTOMY":"RESECTION",
                "BRONCHOSCOPY":"BIOPSY",
                "PULM E BUS":"BIOPSY","RESECTION":"RESECTION"}

allDf=genRes.copy()
allDf["SampleType"]=allDf['Collection_Type'].map(Collection_Map).fillna("UNKNOWN")
allDf['Primary_Met']=allDf['Primary_Met'].fillna('UNKNOWN')

allDf.loc[allDf["Primary_Met"]=='NOT EVALUABLE', "Primary_Met"]= "UNKNOWN"

#%%
yCol='pctPos'
xCol='TrueAngio_GNE'
#%%
markerMap={"BIOPSY":'o',"RESECTION":'s',"UNKNOWN":'x'}
colorMaps={"PRIMARY":'black',"METASTATIC":'red','NOT EVALUABLE':'gray'}
allDf['Marker']=allDf['SampleType'].map(markerMap).fillna(allDf['SampleType'])
allDf["Color"]=allDf['Primary_Met'].map(colorMaps).fillna('gray')
#%% Fig. S6
markers=['o','s','x']
markerSize=10                                                                                                                                     
fontSize=24
plt.figure(figsize=(10,10))
for m in markers:
    dff = allDf[allDf['Marker'] == m]
    plt.scatter(dff[xCol], dff[yCol], c=dff['Color'],marker=m,s=60)
legend_elements = [
                Line2D([0], [0], marker='s', color='w',markeredgecolor='black',
                       label='Resection',markersize=15),
               Line2D([0], [0], marker='o', color='w',markeredgecolor='black',
                      label='Biopsy',markersize=15),
               Line2D([0], [0], marker='X', color='w',markeredgecolor='black',
                      label='Unknown',markersize=15),
               Patch(facecolor='red', 
                     label='Metastatic'),
               Patch(facecolor='black',
                     label='Primary'),
               Patch(facecolor='gray', label='Unknown')]
plt.legend(handles=legend_elements,fontsize=fontSize) 
plt.ylabel(predName,fontsize=fontSize,color=methodColors[predName])                                                                                                                                               
plt.xlabel('RNA Angio Score',fontsize=fontSize,color=methodColors['RNA Angio'])                                                                                                                                                
#plt.title('IMM150: Effect of Sample Preperation and Location',fontsize=fontSize)

saveFile=os.path.join(saveDir,'FigS6.png')                                                                                
#plt.savefig(saveFile,bbox_inches='tight',dpi=300)
