#!/bin/bash

date
hostname -f

# Parameters
IDENTIFIER=1000_RGB_0-16-39
N_PCS=1
MODEL=resnet50
LAYER=L4

# Files
IMAGE_DIR=/dhc/groups/mpws2022cl1/images/heart/png/$IDENTIFIER
INPUT_INDIV_FILE=/dhc/groups/mpws2022cl1/images/heart/png/$IDENTIFIER.csv
EXCLUDED_INDIV_FILE=/dhc/groups/mpws2022cl1/images/heart/png/excluded_indiv.csv
LOG_FILE=job.log

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

OUTPUT_DIR=./output/${IDENTIFIER}_${MODEL_NAME}
# OUTPUT_DIR=/dhc/groups/mpws2022cl1/output/${IDENTIFIER}_${MODEL_NAME}
mkdir -p $OUTPUT_DIR

# Feature condensation (Create embeddings)
feature_condensation=$(sbatch --parsable --output=$OUTPUT_DIR/$LOG_FILE pipeline/feature_condensation.slurm $INPUT_INDIV_FILE $EXCLUDED_INDIV_FILE $OUTPUT_DIR $IMAGE_DIR $N_PCS $MODEL $PRETRAINING $LAYER)

# Run LMM
EMBEDDINGS_FILE=${OUTPUT_DIR}/${MODEL}_${MODEL_NAME}_$LAYER.txt
lmm=$(sbatch --parsable --dependency=afterok:$feature_condensation --output=$OUTPUT_DIR/$LOG_FILE pipeline/lmm.slurm $OUTPUT_DIR $EMBEDDINGS_FILE $N_PCS)

# Plot and change file permissions of output dir
plot=$(sbatch --dependency=afterok:$lmm --output=$OUTPUT_DIR/$LOG_FILE pipeline/plot.slurm $OUTPUT_DIR)

# show dependencies in squeue output:
squeue -u $USER -o "%.8A %.4C %.10m %.20E"