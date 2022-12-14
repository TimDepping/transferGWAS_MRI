#!/bin/bash -eux
#SBATCH --job-name=lmm
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=hpcpu # -p
#SBATCH --cpus-per-task=64 # -c
#SBATCH --mem=64gb
#SBATCH --output=lmm_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

# eval "$(conda shell.bash hook)"
# conda activate transfer_gwas

cd /dhc/home/tim.depping/GitHub/master_project
/dhc/home/tim.depping/conda3/envs/transfer_gwas/bin/python ./lmm/run_lmm.py --config ./lmm/config.toml