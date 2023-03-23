#!/bin/bash -eux
#SBATCH --job-name=lmm_on_prediction
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --time=4320
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=18
#SBATCH --mem=256gb
#SBATCH --output=lmm_on_prediction_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

OUTPUT_DIR=/dhc/groups/mpws2022cl1/output/50000_RGB_0-16-39_imagenet_2023_03_06_11_09_24
LMM_RESULT_DIR=$OUTPUT_DIR/lmm/results_prediction
EMBEDDINGS_FILE=/dhc/groups/mpws2022cl1/images/heart/png/50000_prediction.txt
N_PCS=1
IN_PLINK_BUT_NOT_IMPUTED_FILE=/dhc/groups/mpws2022cl1/images/heart/png/bolt.in_plink_but_not_imputed.FID_IID.80.txt

# Run LMM
python -u lmm/run_lmm.py \
--bed "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bed" \
--bim "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bim" \
--fam "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr1.fam" \
--cov "$OUTPUT_DIR/covariates.txt" \
--cov_cols "sex" "past_tobacco" "assessment_center" "pace_maker" \
--qcov_cols "age" "genet_PC_{1:10}" "geno_batch" \
--emb "$EMBEDDINGS_FILE" \
--first_pc 0 \
--last_pc $((N_PCS-1)) \
--run_imputed \
--bgen "$OUTPUT_DIR/lmm/preprocessing_imp_output/chr{1:22}.bgen" \
--sample "$OUTPUT_DIR/lmm/preprocessing_imp_output/chr1.sample" \
--out_dir $LMM_RESULT_DIR \
--threads $SLURM_JOB_CPUS_PER_NODE \
--remove $IN_PLINK_BUT_NOT_IMPUTED_FILE

# Create mhat and qq plots
python -u lmm/create_plots.py \
--results_dir $LMM_RESULT_DIR \
--output_dir $LMM_RESULT_DIR/plots

# Change permissions for output dir
chgrp -R mpws2022cl1 $LMM_RESULT_DIR
chmod 770 $LMM_RESULT_DIR -R

echo "DONE"