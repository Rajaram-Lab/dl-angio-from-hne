# Data Generation 
This directory has code for generating patches from Whole Slide Images from various datasets. The code uses a WSI image and a corresponding mask and selects random points within the mask region to select patch locations.  The masks were pre-generated using a tumor region classifier and also a small fraction where regions are manually annotated around tumor regions. This code generates patches at 20X magnification.
## Running the code
We use a single [script](Generate_Patches.py) for patch generation for all cohorts. The code can be pointed to run patch generation for a specific cohort by pointing to the corresponding yaml file. Assuming you are running code from this directory within an appropriate Conda encironment, one could extract patches for the IMM150 cohort as follows:

`python Generate_Patches.py --yamlFile ../Params/Generate_Patches_Imm150.yaml`

 The [yaml file](../Params/Generate_Patches_Imm150.yaml) contains information on WSI location, mask location, list of slides to avoid, and location of patches to be saved. Code can be applied to other cohorts by changing the corresponding variables in the yaml file.

## Additional Files
[PatchGen.py](PatchGen.py) contains definitions of various functions including extraction of patches, saving and reading patches to and from a location.

