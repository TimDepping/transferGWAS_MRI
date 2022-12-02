#!/bin/bash -eux
#SBATCH --job-name=lmm
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --container-image=nvidia/cuda:11.4.2-base-ubuntu20.04
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --output=job_test_%j.log # %j is job id
 
date
hostname -f
nproc
nvidia-smi
source activate transfer_gwas

eval "$(conda shell.bash hook)"
conda activate transfer_gwas

python lmm/run_lmm.py --config lmm/config.toml