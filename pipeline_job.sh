#!/bin/bash -eux
#SBATCH --job-name=pipeline
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=32 # -c
#SBATCH --gpus=1
#SBATCH --mem=64gb
#SBATCH --output=job_%j.log # %j is job id
#SBATCH --export=ALL

date
hostname -f

IDENTIFIER=50000_RGB_0-16-39
IMAGE_FILES=/dhc/groups/mpws2022cl1/images/heart/png/$IDENTIFIER
CSV_FILE=/dhc/groups/mpws2022cl1/images/heart/png/$IDENTIFIER.csv
CLEANED_CSV_FILE=/dhc/groups/mpws2022cl1/images/heart/png/${IDENTIFIER}_cleaned.csv
N_PCS=10
MODEL=resnet50
LAYER=L4

# If `imagenet`: load the default pytorch weights
# else: specify path to `.pt` with state dict
# PRETRAINING=imagenet
PRETRAINING=/dhc/groups/mpws2022cl1/models/50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40.pt

if [[ "$PRETRAINING" == "imagenet" ]]
then
MODEL_NAME=imagenet
else
BASE_NAME=$(basename ${PRETRAINING})
MODEL_NAME=${BASE_NAME%.*}
fi

# OUTPUT_DIR=./output/$IDENTIFIER\_$MODEL_NAME\_$SLURM_JOB_ID
OUTPUT_DIR=/dhc/groups/mpws2022cl1/output/$IDENTIFIER\_$MODEL_NAME\_$SLURM_JOB_ID

# Create output dir
mkdir -p $OUTPUT_DIR

# Clean individuals
python -u input/scripts/clean_individuals.py $CSV_FILE PATH_TO_EXCLUDE_CSV 

# Create embeddings
python -u feature_condensation.py $CLEANED_CSV_FILE $OUTPUT_DIR \
--base_img_dir $IMAGE_FILES \
--n_pcs $N_PCS \
--model $MODEL \
--pretraining $PRETRAINING \
--layer $LAYER

# Extract individuals from embeddings file

EMBEDDINGS_FILE=${MODEL}_${MODEL_NAME}_$LAYER.txt
python -c "import pandas as pd; pd.read_csv('$OUTPUT_DIR/${EMBEDDINGS_FILE}', sep=' ')[['FID', 'IID']].to_csv('$OUTPUT_DIR/indiv.txt', sep=' ', header=False, index=False)"

# Preprocess genetic data
sh lmm/preprocessing_ma.sh $OUTPUT_DIR/lmm/preprocessing_ma_output $OUTPUT_DIR/indiv.txt
sh lmm/preprocessing_imp.sh $OUTPUT_DIR/lmm/preprocessing_imp_output $OUTPUT_DIR/indiv.txt $SLURM_JOB_CPUS_PER_NODE

# Preprocess covariates
python -u lmm/preprocessing_cov.py /dhc/projects/ukbiobank/original/phenotypes/ukb_phenotypes.csv \
--indiv $OUTPUT_DIR/indiv.txt \
--output $OUTPUT_DIR/covariates.txt

# Run LMM
python -u lmm/run_lmm.py \
--bed "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bed" \
--bim "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bim" \
--fam "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr1.fam" \
--cov "$OUTPUT_DIR/covariates.txt" \
--cov_cols "sex" \
--INT "" \
--emb "$OUTPUT_DIR/$EMBEDDINGS_FILE" \
--first_pc 0 \
--last_pc $((N_PCS-1)) \
--run_imputed \
--bgen "$OUTPUT_DIR/lmm/preprocessing_imp_output/chr{1:22}.bgen" \
--sample "$OUTPUT_DIR/lmm/preprocessing_imp_output/chr{1:22}.sample" \
--out_dir "$OUTPUT_DIR/lmm/results" \
--threads $SLURM_JOB_CPUS_PER_NODE

# Create mhat and qq plots
python -u lmm/create_plots.py \
--results_dir $OUTPUT_DIR/lmm/results \
--output_dir $OUTPUT_DIR/lmm/results/plots

# Change permissions for output dir
chgrp -R mpws2022cl1 $OUTPUT_DIR
chmod 770 $OUTPUT_DIR -R

date