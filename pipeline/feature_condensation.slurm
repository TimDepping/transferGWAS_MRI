#!/bin/bash -eux
#SBATCH --job-name=feature_condensation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64gb
#SBATCH --export=ALL

INPUT_INDIV_FILE=$1
EXCLUDED_INDIV_FILE=$2
OUTPUT_DIR=$3
IMAGE_DIR=$4
N_PCS=$5
MODEL=$6
PRETRAINING=$7
LAYER=$8
USE_MC=$9

eval "$(conda shell.bash hook)"
conda activate transfer_gwas

CLEANED_INDIV_FILE=${OUTPUT_DIR}/cleaned_indiv.csv

# Create output dir
mkdir -p $OUTPUT_DIR

# Clean individuals
python -u input/scripts/clean_individuals.py $INPUT_INDIV_FILE $EXCLUDED_INDIV_FILE $CLEANED_INDIV_FILE

if $USE_MC
then
python -u feature_condensation.py $CLEANED_INDIV_FILE $OUTPUT_DIR \
--base_img_dir $IMAGE_DIR \
--n_pcs $N_PCS \
--model $MODEL \
--pretraining $PRETRAINING \
--layer $LAYER \
--use_mc
else
python -u feature_condensation.py $CLEANED_INDIV_FILE $OUTPUT_DIR \
--base_img_dir $IMAGE_DIR \
--n_pcs $N_PCS \
--model $MODEL \
--pretraining $PRETRAINING \
--layer $LAYER
fi