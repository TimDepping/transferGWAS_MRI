#!/bin/bash -eux
#SBATCH --job-name=lmm
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=hpcpu
#SBATCH --cpus-per-task=124
#SBATCH --mem=256gb
#SBATCH --output=lmm_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

OUTPUT_DIR=$1
EMBEDDINGS_FILE=$2
N_PCS=$3

eval "$(conda shell.bash hook)"
conda activate transfer_gwas

INDIV_FILE=$OUTPUT_DIR/indiv.txt

# Extract individuals from embeddings
python -c "import pandas as pd; pd.read_csv('$EMBEDDINGS_FILE', sep=' ')[['FID', 'IID']].to_csv('$INDIV_FILE', sep=' ', header=False, index=False)"

# Preprocess genetic data
sh lmm/preprocessing_ma.sh $OUTPUT_DIR/lmm/preprocessing_ma_output $INDIV_FILE
sh lmm/preprocessing_imp.sh $OUTPUT_DIR/lmm/preprocessing_imp_output $INDIV_FILE $SLURM_JOB_CPUS_PER_NODE

# Preprocess covariates
python -u lmm/preprocessing_cov.py /dhc/projects/ukbiobank/original/phenotypes/ukb_phenotypes.csv \
--indiv $INDIV_FILE \
--output $OUTPUT_DIR/covariates.txt

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
--bgen "$OUTPUT_DIR/lmm/preprocessing_imp_output/chr{1:22}.bgen" \
--sample "$OUTPUT_DIR/lmm/preprocessing_imp_output/chr{1:22}.sample" \
--out_dir "$OUTPUT_DIR/lmm/results" \
--threads $SLURM_JOB_CPUS_PER_NODE
# Add 'remove' paramenter if there is an error regarding missing data e.g.
# --remove "${OUTPUT_DIR}/lmm/preprocessing_imp_output/bolt.in_plink_but_not_imputed.FID_IID.80.txt"