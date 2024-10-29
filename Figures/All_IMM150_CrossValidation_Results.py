#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 07:35:11 2023
 File used for printing output for 3 Fold Cross Validation results on IMM150 data
 In addition the Effect of Grade, 'MTZRGR' column on treatment response is printed out

"""
survivalThresholdMethod='IMM150_Median' #'TCGA'
assert survivalThresholdMethod in ['TCGA','IMM150_Median']
import numpy as np
import os
import pandas as pd

# PLEASE CHANGE THIS TO POINT TO YOUR CODE DIR
codeDir=os.path.dirname(__file__) 
import scipy.stats as stats
from sklearn import metrics
import yaml
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import rankdata
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

# %% Load up Various Params
globalParamsFile=os.path.join(codeDir,'Global_Params.yml')
globalParams=yaml.safe_load(open(globalParamsFile,'r'))
saveDir=codeDir
methodColors=globalParams['Plotting']['methodColors']
predName=globalParams['Plotting']['methodNames']['H&E Prediction']


armRenamingDict=globalParams['Plotting']['IMM150']['armNames']

#%% Load Up Saved IMM150 Results
# below file includes all TrueRNA data and 3 fold predictions
saveFileCsv=os.path.join(codeDir,'All_IMM150_6.csv')
allRes=pd.read_csv(saveFileCsv)
metricList=['Angio', 'maskAngio', 'pctPos']
allRes['CD31']=[np.float32(s) if is_number(s) else np.NAN for s in allRes['%_CD31']]
allRes['TrueAngio']=allRes['TrueAngio_GNE']
allRes=allRes.rename({'TrueAngio':'RNA_Angio'},axis=1)

exclude_list=globalParams['Data']['IMM150']['failedQcFiles']
duplicate_list= globalParams['Data']['IMM150']['duplicateFiles']
removeList=exclude_list+duplicate_list
genentechRes=allRes[~allRes.FILENAME.isin(removeList)]
genentechRes=genentechRes.reset_index()
genentechRes['FUHRMAN']=genentechRes['FUHRMAN'].fillna('.')
genentechRes['Grade']=[np.NAN if '.' in g else np.float32(g[-1]) for g in genentechRes['FUHRMAN']]
genentechRes['MTZRGR']=[np.float32(s) if is_number(s) else np.NAN for s in genentechRes['MTZRGR']]
grpMap={'LOW': 0, 'INTERMEDIATE': 1, 'HIGH':2}
genentechRes['MSKCC']=genentechRes['IVRMTZR'].map(grpMap)
genentechRes['MSKCC1']=genentechRes['MSKCC']
#%% loop through 3 folds and print results for each fold
for foldNum in [0,1,2]:
    
    print('**********  Results for Fold ********: ', foldNum)
    colName='pctPos_Fold_'+str(foldNum)
    genentechRes['pctPos']=genentechRes[colName]
# Get Cutoffs (currently set at median) for Low/High Angio Call 

    fieldCutoffs={}
    fieldCutoffs['RNA_Angio']=genentechRes['RNA_Angio'].median()
    if survivalThresholdMethod=='TCGA':
        fieldCutoffs['pctPos']=5.66
    else:
        fieldCutoffs['pctPos']=genentechRes['pctPos'].median()
    
    fieldCutoffs['CD31']=genentechRes['CD31'].median()
    fieldCutoffs['Grade']=2
    fieldCutoffs['MTZRGR']=1
    fieldCutoffs['MSKCC']=0
    fieldCutoffs['MSKCC1']=1
    aucs=[]
    strs=[]


    print('******* Hazard Ratio calculations for Fold ** ', foldNum)
    fieldsToAnalyze=['RNA_Angio','pctPos','CD31','Grade','MTZRGR','MSKCC','MSKCC1']
    fieldNameSimpleDict={'RNA_Angio':'RNA Angio',
                        'pctPos': predName,'CD31':'CD31',
                        'Grade':'Grade','MTZRGR':'MTZRGR',
                        'MSKCC': 'MSKCC', 'MSKCC1': 'MSKCC1'}
    labelfontSize=12
    fontSize=18
    
    armList=list(armRenamingDict)
    armsToPlot=['SUNITINIB']
    
    readOutList=['Method','Arm','N_high','N_low','HR','HR_95Low','HR_95High','pVal','CI']
    hazardDfCombined={r:[]for r in readOutList}
    
    
    for armCounter,armName in enumerate(armList):
        
        armRes=genentechRes[genentechRes['ACTARM']==armName].copy()
             
        for field in fieldsToAnalyze:
            fieldNameSimple=fieldNameSimpleDict[field]
    
            isGood=np.logical_and(np.logical_not(pd.isnull(armRes[field].values)),
                                  np.logical_not(pd.isnull(armRes['PFSINV'].values)))
            fieldThresholds={field:lambda x: x> fieldCutoffs[field]}
            
            
            survDf=armRes[isGood].copy()
            survDf['PFSINV']=survDf['PFSINV']/30.4  # convert days to months
            isHigh=fieldThresholds[field](survDf[field].values)
            
            # Cox Proportional Hazards Calculation
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

        
    hazardDfCombined=pd.DataFrame(hazardDfCombined)
    pd.set_option('display.max_columns', None)

    print(hazardDfCombined.to_string(index=False, header=True))

#  Objective Response Rate calculations
    print('*** Objective response results for fold number **', foldNum)
    readOutList=['Method','Arm','N_high','N_low','CR_Frac_H','PR_Frac_H','Resp_Frac_H','CR_Frac_L','PR_Frac_L','Resp_Frac_L']
    respDfCombined={r:[]for r in readOutList}
    
    
    armColors=globalParams['Plotting']['IMM150']['armColors']
    
    barWidth=0.75 # thickness of each bar
    catWidth=1 #spacing between Low/High bars in each arm
    groupWidth=2.5 #spacing between arms
    #groupLabelYPos=-0.075 # yPosition of the 'High N=xxx' label
    #armLabelYPos=-0.115 #yPosition of the arm label e.g. 'Sunitinib'
    groupLabelYPos=-0.080 # yPosition of the 'High N=xxx' label
    armLabelYPos=-0.120 #yPosition of the arm label e.g. 'Sunitinib'
    for field in fieldsToAnalyze: # Loop over readout e.g., RNA_Angio or Mask_Angio
    
        #plt.figure()
       
        isGood=np.logical_and(np.logical_not(pd.isnull(genentechRes[field].values)),
                              np.logical_not(pd.isnull(genentechRes['BESRSPI'].values)))
        
        survDf=genentechRes[isGood]
    
        for armCounter,arm in enumerate(armRenamingDict): # Loop over arm i.e. drug treatment
            armNameSimple=armRenamingDict[arm]
            armDf=survDf[survDf['ACTARM']==arm]
            respDfCombined['Method'].append(field)
            respDfCombined['Arm'].append(armNameSimple)
            for highCounter,isHigh in enumerate([True,False]): # Loop over High/Low level of readout
    
                isInGroup=(armDf[field]>fieldCutoffs[field])==isHigh
                groupDf=armDf[isInGroup]
                nGroup=groupDf.shape[0]
                if isHigh:
                    respDfCombined['N_high'].append(nGroup)
                else:
                    respDfCombined['N_low'].append(nGroup)
                cumFrac=0
                x=groupWidth*armCounter+(highCounter*catWidth)
    
                for responseCounter,response in enumerate(['CR','PR']): # Loop response level
    
                    nResp=np.sum(groupDf['BESRSPI']==response)
                    frac=nResp/nGroup
                    
                    cumFrac+=frac
                    frac=np.round(frac,3)
                    cumFrac=np.round(cumFrac,3)
                    if response == 'CR':
                        if isHigh:
                            respDfCombined['CR_Frac_H'].append(frac)
                        else:
                            respDfCombined['CR_Frac_L'].append(frac)
                    else:
                        if isHigh:
                            respDfCombined['PR_Frac_H'].append(frac)
                            respDfCombined['Resp_Frac_H'].append(cumFrac)
                        else:
                            respDfCombined['PR_Frac_L'].append(frac)
                            respDfCombined['Resp_Frac_L'].append(cumFrac)
                            
                groupLabel=['High','Low'][highCounter]+'\nn='+str(nGroup)

    respDfCombined=pd.DataFrame(respDfCombined)
    pd.set_option('display.max_columns', None)
    
    print(respDfCombined.to_string(index=False, header=True))
    
    sunitinibRes=genentechRes[genentechRes['ACTARM']=='SUNITINIB'].copy()
    sunitinibRes=sunitinibRes.rename(columns={'BESRSPI':'Response_Level'})
    sunitinibRes=sunitinibRes[~pd.isnull(sunitinibRes['Response_Level'])]
    responderLabels=['CR','PR']
    isResponder=np.array([s in responderLabels for s in sunitinibRes['Response_Level']])
    
# print aucs
    
    print('********Aucs for responders ** For Fold Num ', foldNum)
    for fieldCounter,field in enumerate(fieldsToAnalyze):
        fieldNameSimple=fieldNameSimpleDict[field]
        isGood=np.isfinite(sunitinibRes[field].values)
    
        fpr, tpr, thresholds = metrics.roc_curve(isResponder[isGood], 
                                                 sunitinibRes[field].values[isGood])
        auc=metrics.roc_auc_score(isResponder[isGood], sunitinibRes[field].values[isGood])
        aucs.append(np.round(auc,2))
        strs.append(fieldNameSimple+ ' ='+str(np.round(auc,2)))
    
    for (s, a) in zip(strs, aucs):   
        print(s)


# %%

def CleanVals(data):
    data1=[np.array(x).flatten() for x in data]
    assert np.all([len(x)==len(data1[0]) for x in data1])
    isGood=np.isfinite(data1[0])
    for x in data1:
        isGood=np.logical_and(np.isfinite(x),isGood)
    return [x[isGood] for x in data1]

# %%


armRes=genentechRes[genentechRes['ACTARM']=='SUNITINIB'].copy()
fieldTransforms={
                  'PDL1_IC_Category':lambda s: np.float32(s[-1]) if isinstance(s,str) else np.NAN,
                  'PDL1_TC_Category':lambda s: np.float32(s) if is_number(s) else np.NAN}
for field in fieldTransforms:
   armRes[field]=[fieldTransforms[field](s) for s in armRes[field]]

# %%  print C-index values for various combinations
for foldNum in [0,1,2]:
    
    print('**********  Results for Fold ********: ', foldNum)
    field1='pctPos_Fold_'+str(foldNum)
    
    fieldList=['MSKCC1','Grade','MTZRGR','PDL1_TC_Category','PDL1_IC_Category']
    
    f1CIndex=concordance_index(*CleanVals([armRes['PFSINV'],armRes[field1],armRes['PFSINV.event']]))
    print('***** Combined C-Index *****')
    print('H&E:',np.round(f1CIndex,3))
    for field2 in fieldList:
    
        
        f2CIndex=concordance_index(*CleanVals([armRes['PFSINV'],-armRes[field2],armRes['PFSINV.event']]))
        print(field2+ ' :',np.round(f2CIndex,3))
        
        
        vals1=armRes[field1]
        vals2=armRes[field2]
        isGood=np.logical_and(np.isfinite(vals1),np.isfinite(vals2))
        
        ranks1=rankdata(vals1[isGood])
        ranks2=rankdata(-vals2[isGood])
        ranks=(ranks1+ranks2)/2
        
        combCIndex=concordance_index(*CleanVals([armRes['PFSINV'][isGood],ranks,armRes['PFSINV.event'][isGood]]))
        print(field2+ '+H&E:',np.round(combCIndex,3))
    
