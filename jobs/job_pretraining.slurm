#!/bin/bash -eux
#SBATCH --job-name=pretraining
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=siefermann.laurin@gmail.com
#SBATCH --partition=gpupro # -p
#SBATCH --gpus=1
#SBATCH --output=pretraining_ci_norm_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

cd /dhc/home/laurin.siefermann/masterProject/master_project/pretraining

/dhc/home/laurin.siefermann/conda3/envs/transfer_gwas/bin/python -u \
/dhc/home/laurin.siefermann/masterProject/master_project/pretraining/training_main.py /dhc/groups/mpws2022cl1/images/heart/png/50000_RGB_0-16-39 /dhc/groups/mpws2022cl1/input/cardiacIndex_normalized_38774.csv --size 224 --model_name 50_ci

date