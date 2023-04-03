# Genome-wide association studies of MRI heart imaging phenotypes
For our masterproject 2022/23 (Digital Health & Machine Learning, Prof. Dr. Christoph Lippert, HPI ) we applied transferGWAS (https://github.com/mkirchler/transferGWAS) to MRI heart images from the UK Biobank. transferGWAS is a method for performing genome-wide association studies on whole images, leveraging Deep Learning. In our project, we adapted the work of Matthias Kirchler to work with 2.5D MRI image modalities. This means we process 2D images plus as well as an additional "time-dimension". 

This repository provides the code we used to run our transferGWAS on UK Biobank data. transferGWAS has 3 steps: 1. pretraining, 2. feature condensation, and 3. LMM association analysis. 

* **`pretraining`:** provides code for training our models on heart mri images (3-Channel RGB Images, 50-Channel Images (Tensors)). 

* **`lmm`:** this part is a wrapper for the BOLT-LMM association analysis.

* **`feature_condensation`:** provides code to go from trained model to low-dimensional condensed features

* **`models`:** TODO: I think we need to add this, right?

* **`pipeline`:** All you need to automatically run feature condensation, the LMM analysis and to create plots to visualize the results.

* **`utils`:** Some helper scripts to prepare and create inputs for training and feature condensation.

## Getting started
Start by cloning this repo:
```bash
git clone https://github.com/TimDepping/master_project
```

### Python
All deep learning parts are built in pytorch. We recommend using some up-to-date version of anaconda and then creating the environments from the yml files.

To run the feature_condensation and the LMM you will need to install all dependencies from the `environment.yml` file.

```bash
conda env create --file environment.yml
conda activate transfer_gwas
```

If you want to train your own models you will need to install all dependencies from the `environment_train.yml` file.
During our project we discovered that we need a newer torch version to work with 2.5D data, which was not compatible with our initial environment. Therefore we created a new environment with the updated torch version.

```bash
conda env create --file environment_train.yml
conda activate torch_update
```