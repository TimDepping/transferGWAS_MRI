#!/bin/bash -eux
#SBATCH --job-name=pretraining
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=siefermann.laurin@gmail.com
#SBATCH --partition=gpupro # -p
#SBATCH --gpus=1
#SBATCH --output=createTensor50k%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

cd /dhc/home/laurin.siefermann/masterProject/master_project/pretraining

/dhc/home/laurin.siefermann/conda3/envs/transfer_gwas/bin/python -u \
/dhc/home/laurin.siefermann/masterProject/master_project/input/scripts/create_multichannel_tensors.py /dhc/groups/mpws2022cl1/images/heart/png/50000_GRAY /dhc/groups/mpws2022cl1/tensor/50000_GRAY

date

