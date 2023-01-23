#!/bin/bash -eux
#SBATCH --job-name=lmm_imputed
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=hpcpu # -p
#SBATCH --cpus-per-task=28 # -c
#SBATCH --mem=256gb
#SBATCH --output=lmm_imputed_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

OUTPUT_DIR= # e.g. /dhc/groups/mpws2022cl1/output/50000_RGB_0-16-39_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40
EMBEDDINGS_FILE= # e.g. /dhc/groups/mpws2022cl1/output/50000_RGB_0-16-39_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40/resnet50_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40_L4.txt
N_PCS=10

# Run LMM
python -u lmm/run_lmm.py \
--bed "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bed" \
--bim "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bim" \
--fam "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr1.fam" \
--cov "$OUTPUT_DIR/covariates.txt" \
--cov_cols "sex, past_tobacco, assessment_center, pace_maker, geno_batch, heart_rate, age, bmi" \
--qcov_cols "age, genet_PC_{1:41}",
--INT "" \
--emb "$EMBEDDINGS_FILE" \
--first_pc 0 \
--last_pc $((N_PCS-1)) \
--run_imputed \
--bgen "${OUTPUT_DIR}/lmm/preprocessing_imp_output/chr{1:22}.bgen" \
--sample "${OUTPUT_DIR}/lmm/preprocessing_imp_output/chr1.sample" \
--out_dir "$OUTPUT_DIR/lmm/results" \
--remove "${OUTPUT_DIR}/lmm/preprocessing_imp_output/bolt.in_plink_but_not_imputed.FID_IID.80.txt"
--threads $SLURM_JOB_CPUS_PER_NODE