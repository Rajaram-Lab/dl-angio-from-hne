
# Model Evaluation
This folder contains the code to evaluate the following four classes of models:

1. Mixed Model (H&E DL Angioscore): the primary model of this work, which predicts the Angioscore and a Vascular Mask given an H&E image 
2. ResNet Based Models: A set of models to predict the Angioscore from patch data, with different parts of the model frozen.
3. CAMIL: Angioscore prediction from pre-extracted patch features using a multiple instance learning paradigm. 
4. UNet: A segmentation model to predict the vascular mask.

## Evaluation Data
 Models were evaluated on patches (extracted as described [here](../Data_Generation/README.md)) from the following datasets:
1. TCGA with matched AngioScores: This is the held-out portion of  with one Whole Slide Image (WSI) per patient
2. IMM150 Data with AngioScores and Response to Treatment Data: This is the IMMotion-150 clinical trial dataset with one WSI per patient
3. UTSeq Data: This dataset has multiple samples per slide where RNA expression data is available
4. TMA Data with Grade, mutation and other prognostic data: This dataset has multiple Tissue microarray punches per patient.
5. UTSW response to Treatment data: This data has Time to Next Treatment (TNT) data for patients undergoing VEGF treatment.

Note: Prior to inference, patches for all datasets except TCGA (which was used in training) stain normalization was performed against TCGA [as previously described](https://github.com/biototem/cancres-2022-intratumoral-heterogeneity-dl-paper/blob/master/Util/NormalizationCore.py).


## Evaluating Mixed, Resnet and UNet Models 
The three sets of models  can be evaluated as follows:
1. Launch the appropriate Conda environment from this directory 
2. Set parameters like file save paths etc in the corresponding yaml files for the [Mixed](/Params/Eval_MixedModel_TCGA.yaml), [Resnet](/Params/Eval_ResNetModel_TCGA.yaml), [UNet](/Params/Eval_UNetModel_TCGA.yaml) models
3. To perform evaluation on the whole slide image cohorts (TCGA, IMM150, UTSW Response)  point to the corresponding yaml file: 

	`python Eval_MixedModel_WSI.py --yamlFile ../Params/Eval_MixedModel_TCGA.yaml`  

	`python Eval_ResNetModel_WSI.py --yamlFile ../Params/Eval_ResNetModel_TCGA.yaml`    

	`python Eval_UNetModel_WSI.py --yamlFile ../Params/Eval_UNetModel_TCGA.yaml`  

4. To perform evaluation on the TMA and UTSeq Cohorts use the corresponding Evaluation scripts. For example to evaluate our mixed model:

   `python Eval_MixedModel_TMA.py --yamlFile ../Params/Eval_MixedModel_TMA.yaml`  

   `python Eval_ResNetModel_UTSeq.py --yamlFile ../Params/Eval_ResNetModel_UTSeq.yaml`    

   

## Evaluation of [CAMIL](https://github.com/KatherLab/marugoto) Model

The [CAMIL](https://www.nature.com/articles/s41467-024-45589-1) model was trained using code from their [github repository](https://github.com/KatherLab/marugoto) as described [here](../Model_Training/README.md).  To keep consistency with our training and evaluation of other models we 1) Use the same normalized patches extracted from Tumor areas at 20X magnification as for the other methods, 2) Use ResNet50 model for generating features per patch. 

