#!/bin/bash

date
hostname -f

# Parameters
IDENTIFIER=50000_RGB_0-16-39
N_PCS=10
MODEL=resnet50
LAYER=L4

# Files
IMAGE_DIR=/dhc/groups/mpws2022cl1/images/heart/png/$IDENTIFIER
INPUT_INDIV_FILE=/dhc/groups/mpws2022cl1/images/heart/png/$IDENTIFIER.csv
EXCLUDED_INDIV_FILE=/dhc/groups/mpws2022cl1/images/heart/png/excluded_indiv_2.csv

# This file needs to be adapted if we change the images
IN_PLINK_BUT_NOT_IMPUTED_FILE=/dhc/groups/mpws2022cl1/images/heart/png/bolt.in_plink_but_not_imputed.FID_IID.80.txt

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

# OUTPUT_DIR=./output/${IDENTIFIER}_${MODEL_NAME}
CURRENT_DATE=$(date +'%Y_%m_%d_%H_%M_%S')
OUTPUT_DIR=/dhc/groups/mpws2022cl1/output/${IDENTIFIER}_${MODEL_NAME}_${CURRENT_DATE}
LOG_DIR=$OUTPUT_DIR/log
mkdir -p $OUTPUT_DIR
mkdir $LOG_DIR

# Feature condensation (Create embeddings)
feature_condensation=$(sbatch --parsable --output=$LOG_DIR/feature_condensation.log pipeline/feature_condensation.slurm $INPUT_INDIV_FILE $EXCLUDED_INDIV_FILE $OUTPUT_DIR $IMAGE_DIR $N_PCS $MODEL $PRETRAINING $LAYER)

# Run LMM
EMBEDDINGS_FILE=${OUTPUT_DIR}/${MODEL}_${MODEL_NAME}_$LAYER.txt
lmm=$(sbatch --parsable --dependency=afterok:$feature_condensation --output=$LOG_DIR/lmm.log pipeline/lmm.slurm $OUTPUT_DIR $EMBEDDINGS_FILE $N_PCS $IN_PLINK_BUT_NOT_IMPUTED_FILE)

# Plot and change file permissions of output dir
plot=$(sbatch --dependency=afterok:$lmm --output=$LOG_DIR/plot.log pipeline/plot.slurm $OUTPUT_DIR)

# show dependencies in squeue output:
squeue -u $USER