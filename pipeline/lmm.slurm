#!/bin/bash -eux
#SBATCH --job-name=lmm
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --time=4320
#SBATCH --partition=hpcpu,vcpu,gpua100,gpupro,gpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=256gb
#SBATCH --export=ALL
 
date
hostname -f

OUTPUT_DIR=$1
EMBEDDINGS_FILE=$2
N_PCS=$3
IN_PLINK_BUT_NOT_IMPUTED_FILE=$4

eval "$(conda shell.bash hook)"
conda activate transfer_gwas

INDIV_FILE=$OUTPUT_DIR/indiv.txt

# Extract individuals from embeddings
python -c "import pandas as pd; pd.read_csv('$EMBEDDINGS_FILE', sep=' ')[['FID', 'IID']].to_csv('$INDIV_FILE', sep=' ', header=False, index=False)"

# Preprocess genetic data
sh lmm/preprocessing_ma.sh $OUTPUT_DIR/lmm/preprocessing_ma_output $INDIV_FILE
sh lmm/preprocessing_imp.sh $OUTPUT_DIR/lmm/preprocessing_imp_output $INDIV_FILE $SLURM_JOB_CPUS_PER_NODE

# Preprocess covariates
python -u lmm/preprocessing_cov.py /dhc/projects/ukbiobank/original/phenotypes/ukb49727.csv \
--indiv $INDIV_FILE \
--output $OUTPUT_DIR/covariates.txt

# Run LMM
if [ "$IN_PLINK_BUT_NOT_IMPUTED_FILE" ];then
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
--bgen "${OUTPUT_DIR}/lmm/preprocessing_imp_output/chr{1:22}.bgen" \
--sample "${OUTPUT_DIR}/lmm/preprocessing_imp_output/chr1.sample" \
--out_dir "$OUTPUT_DIR/lmm/results" \
--threads $SLURM_JOB_CPUS_PER_NODE \
--remove $IN_PLINK_BUT_NOT_IMPUTED_FILE
else
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
--bgen "${OUTPUT_DIR}/lmm/preprocessing_imp_output/chr{1:22}.bgen" \
--sample "${OUTPUT_DIR}/lmm/preprocessing_imp_output/chr1.sample" \
--out_dir "$OUTPUT_DIR/lmm/results" \
--threads $SLURM_JOB_CPUS_PER_NODE
fi