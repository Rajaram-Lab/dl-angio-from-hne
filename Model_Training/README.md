# Model Training
This folder contains the code to train the following four classes of models:

1. Mixed Model (H&E DL Angioscore): the primary model of this work, which predicts the Angioscore and a Vascular Mask given an H&E image 
2. UNet: A segmentation model to predict the vascular mask.
3. ResNet Based Models: A set of models to predict the Angioscore from patch data, with different parts of the model frozen.
4. CAMIL: Angioscore prediction from pre-extracted patch features using a multiple instance learning paradigm. 

## Training Data
The training data for all model consists of some subset of the patches:
1. From TCGA with matched AngioScores: Used by Mixed Model, ResNet, CAMIL
2. From UTSW with matched Vascular Masks: Used by Mixed Model and UNet

## Training Mixed, Resnet and UNet Models
We use custom code for these three sets of models which can be run as follows:
1. Launch the appropriate Conda environment from this directory 
2. Set parameters like file save paths etc in the corresponding yaml files for the [Mixed](/Params/Train_MixedModel.yaml), [Resnet](/Params/Train_ResNetModel.yaml), [UNet](/Params/Train_UNetModel.yaml) models
3. Run the training code by pointing to the corresponding yaml file. 

`python Train_MixedModel.py --yamlFile ../Params/Train_MixedModel.yaml`

`python Train_ResNetModel.py --yamlFile ../Params/Train_ResNetModel.yaml`

`python Train_UNetModel.py --yamlFile ../Params/Train_UNetModel.yaml`

Additional files in this folder include:
1. [MixedModel.py](MixedModel.py): provides the definition of the model.
2. [Dice.py](Dice.py): Contains a definition of the Dice Loss
3. [extendedImgaugAugmenters.py](extendedImgaugAugmenters.py): File used for augmenting data while training various models.

## Training [CAMIL](https://github.com/KatherLab/marugoto) Model
We used publicly available Contrastively Clustered attention-based Multiple Instance learning model as described here [CAMIL](https://www.nature.com/articles/s41467-024-45589-1). We used the  procedure outlined in the [github repository](https://github.com/KatherLab/marugoto)  for generating features and training the model, with the following changes both of which lead to improvement in performance: 1) Use ResNet50 model for generating features instead of default approach. 2) Use patches extracted from Tumor areas at 20X magnification instead of default method.
