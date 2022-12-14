#!/bin/bash -eux
#SBATCH --job-name=image_preprocessing
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=hpcpu # -p
#SBATCH --cpus-per-task=32 # -c
#SBATCH --mem=32gb
#SBATCH --output=image_preprocessing_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

# eval "$(conda shell.bash hook)"
# conda activate transfer_gwas

cd /dhc/home/tim.depping/GitHub/master_project
/dhc/home/tim.depping/conda3/envs/transfer_gwas/bin/python /dhc/home/tim.depping/GitHub/master_project/image_preprocessing.py --output_dir /dhc/groups/mpws2022cl1/images/heart/png/100_RGB_0-16-39 --number_of_patients 100 --create_zip True
chgrp -R mpws2022cl1 /dhc/groups/mpws2022cl1/images
chmod 770 /dhc/groups/mpws2022cl1/images -R