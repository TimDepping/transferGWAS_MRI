#!/bin/bash -eux
#SBATCH --job-name=pretraining
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=siefermann.laurin@gmail.com
#SBATCH --partition=gpupro # -p
#SBATCH --gpus=1
#SBATCH --output=pretraining_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

# eval "$(conda shell.bash hook)"
# conda activate transfer_gwas

# NUMBER_OF_PATIENTS="50000"
# NAME="${NUMBER_OF_PATIENTS}_RGB_0-16-39"

cd /dhc/home/laurin.siefermann/masterProject/master_project/pretraining

/dhc/home/laurin.siefermann/conda3/envs/transfer_gwas/bin/python -u \
/dhc/home/laurin.siefermann/masterProject/master_project/pretraining/training_main.py /dhc/groups/mpws2022cl1/images/heart/png/50000_RGB_0-16-39 /dhc/groups/mpws2022cl1/input/ejectionFraction_38781.csv --size 224\

date