#!/bin/bash -eux
#SBATCH --job-name=image_preprocessing
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=hpcpu # -p
#SBATCH --cpus-per-task=1 # -c
#SBATCH --mem=16gb
#SBATCH --output=image_preprocessing_%j.log # %j is job id
#SBATCH --export=ALL
 
date
hostname -f

# eval "$(conda shell.bash hook)"
# conda activate transfer_gwas

NUMBER_OF_PATIENTS="50000"
NAME="${NUMBER_OF_PATIENTS}_RGB_0-16-39"

echo $NAME

cd /dhc/home/tim.depping/GitHub/master_project

/dhc/home/tim.depping/conda3/envs/transfer_gwas/bin/python -u \
/dhc/home/tim.depping/GitHub/master_project/image_preprocessing.py \
--temp_dir ./.tmp_$NAME \
--output_dir /dhc/groups/mpws2022cl1/images/heart/png/$NAME \
--number_of_patients $NUMBER_OF_PATIENTS \
--create_zip True

chgrp -R mpws2022cl1 /dhc/groups/mpws2022cl1/images
chmod 770 /dhc/groups/mpws2022cl1/images -R

date