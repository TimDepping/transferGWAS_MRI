#!/bin/bash -eux
#SBATCH --job-name=lmm_on_label
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --time=2880
#SBATCH --partition=hpcpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=256gb
#SBATCH --output=lmm_on_label_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

OUTPUT_DIR=/dhc/groups/mpws2022cl1/output/50000_RGB_0-16-39_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40_label # e.g. /dhc/groups/mpws2022cl1/output/50000_RGB_0-16-39_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40
EMBEDDINGS_FILE=/dhc/groups/mpws2022cl1/output/50000_RGB_0-16-39_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40_label/embeddings.txt # e.g. /dhc/groups/mpws2022cl1/output/50000_RGB_0-16-39_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40/resnet50_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40_L4.txt
N_PCS=1

# Run LMM
python -u lmm/run_lmm.py \
--bed "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bed" \
--bim "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bim" \
--fam "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr1.fam" \
--cov "$OUTPUT_DIR/covariates.txt" \
--cov_cols "sex" "past_tobacco" "assessment_center" "pace_maker" \
--qcov_cols "age" "genet_PC_{1:40}" "geno_batch" "heart_rate" "bmi" \
--emb "$EMBEDDINGS_FILE" \
--first_pc 0 \
--last_pc $((N_PCS-1)) \
--run_imputed \
--bgen "${OUTPUT_DIR}/lmm/preprocessing_imp_output/chr{1:22}.bgen" \
--sample "${OUTPUT_DIR}/lmm/preprocessing_imp_output/chr1.sample" \
--out_dir "$OUTPUT_DIR/lmm/results" \
--threads $SLURM_JOB_CPUS_PER_NODE \
--remove "/dhc/groups/mpws2022cl1/output/50000_RGB_0-16-39_50_ef_norm_wo_outliers_ae_2023_01_10_08_14_40_label/lmm/preprocessing_imp_output/bolt.in_plink_but_not_imputed.FID_IID.80.txt"

echo "DONE"