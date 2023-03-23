#!/bin/bash

# This script runs the feature condensation, the LMM analysis and creates plots to visualize the results.

# Print the current date and hostname
date
hostname -f

# Set Parameters
IDENTIFIER=50000_GRAY # For GRAY 50 Channel images, IDENTIFIER=50000_RGB_0-16-39 for RGB 3 Channel images
N_PCS=10
MODEL=resnet50
LAYER=L4
USE_MC=true

# Set file paths
IMAGE_DIR=/dhc/groups/mpws2022cl1/tensor/$IDENTIFIER
INPUT_INDIV_FILE=$IMAGE_DIR.csv
EXCLUDED_INDIV_FILE=/dhc/groups/mpws2022cl1/images/heart/png/excluded_indiv_2.csv
IN_PLINK_BUT_NOT_IMPUTED_FILE=/dhc/groups/mpws2022cl1/images/heart/png/bolt.in_plink_but_not_imputed.FID_IID.80.txt # If this file does not exist yet, set this variable to ""
PRETRAINING=/dhc/groups/mpws2022cl1/models/50_minMaxScaling_ef_mc_2023_02_16_15_53_44.pt # For 50 Channel
# Other options for PRETRAINING:
# PRETRAINING=imagenet # For imagenet
# PRETRAINING=/dhc/groups/mpws2022cl1/models/50_minMaxScaling_ef__ae_2023_02_13_17_54_24.pt # For 3 Channel images

if [[ "$PRETRAINING" == "imagenet" ]]; then
  MODEL_NAME=imagenet
else
  BASE_NAME=$(basename ${PRETRAINING})
  MODEL_NAME=${BASE_NAME%.*}
fi

# Set directories
CURRENT_DATE=$(date +'%Y_%m_%d_%H_%M_%S')
OUTPUT_DIR=/dhc/groups/mpws2022cl1/output/${IDENTIFIER}_${MODEL_NAME}_${CURRENT_DATE}
LOG_DIR=$OUTPUT_DIR/log

# Create directories
mkdir -p $OUTPUT_DIR
mkdir $LOG_DIR

# Log file
PIPELINE_LOG=$LOG_DIR/pipeline.log

# Run feature condensation and get the job id
FEATURE_CONDENSATION_CMD="sbatch --parsable --output=$LOG_DIR/feature_condensation.log pipeline/feature_condensation.slurm $INPUT_INDIV_FILE $EXCLUDED_INDIV_FILE $OUTPUT_DIR $IMAGE_DIR $N_PCS $MODEL $PRETRAINING $LAYER $USE_MC"
echo $FEATURE_CONDENSATION_CMD >> $PIPELINE_LOG
feature_condensation=$($FEATURE_CONDENSATION_CMD)

# Run LMM and get the job id
EMBEDDINGS_FILE=${OUTPUT_DIR}/${MODEL}_${MODEL_NAME}_$LAYER.txt
LMM_CMD="sbatch --parsable --dependency=afterok:$feature_condensation --output=$LOG_DIR/lmm.log pipeline/lmm.slurm $OUTPUT_DIR $EMBEDDINGS_FILE $N_PCS \"$IN_PLINK_BUT_NOT_IMPUTED_FILE\""
echo $LMM_CMD >> $PIPELINE_LOG
lmm=$($LMM_CMD)

# Run plot and change file permissions of output dir
PLOT_CMD="sbatch --dependency=afterok:$lmm --output=$LOG_DIR/plot.log pipeline/plot.slurm $OUTPUT_DIR"
echo $PLOT_CMD >> $PIPELINE_LOG
$PLOT_CMD

# show dependencies in squeue output:
squeue -u $USER