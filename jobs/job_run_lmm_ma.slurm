#!/bin/bash -eux
#SBATCH --job-name=lmm
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=hpcpu # -p
#SBATCH --cpus-per-task=64 # -c
#SBATCH --mem=80gb
#SBATCH --output=lmm_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

N_PCS=3

OUTPUT_DIR=/dhc/home/tim.depping/GitHub/master_project/output

# Run LMM
python -u lmm/run_lmm.py \
--bed "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bed" \
--bim "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr{1:22}.bim" \
--fam "$OUTPUT_DIR/lmm/preprocessing_ma_output/chr1.fam" \
--cov "$OUTPUT_DIR/covariates.txt" \
--cov_cols "sex" \
--INT "" \
--emb "$OUTPUT_DIR/resnet50_imagenet_L4.txt" \
--first_pc 0 \
--last_pc $((N_PCS-1)) \
--out_dir "$OUTPUT_DIR/lmm/results" \
--threads $SLURM_JOB_CPUS_PER_NODE