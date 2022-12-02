# Activate conda environment
conda activate transfer_gwas

# Create embeddings
python feature_condensation.py input/image_files/heart_image_files/heart_image_files_all.csv ./out_dir/ --base_img_dir input/image_files/heart_image_files/heart_image_files_all --n_pcs 10

# Extract individuals from embeddings file
python -c "import pandas as pd; pd.read_csv('output/resnet50_imagenet_L4.txt', sep=' ')[['FID', 'IID']].to_csv('output/indiv.txt', sep=' ', header=False, index=False)"

# Preprocess genetic data
sh lmm/preprocessing_ma.sh

# Preprocess covariates
python lmm/preprocessing_cov.py /dhc/projects/ukbiobank/original/phenotypes/ukb_phenotypes.csv --indiv output/indiv.txt --output output/covariates.txt

# Run LMM
python lmm/run_lmm.py --config lmm/config.toml