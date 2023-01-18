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

NUMBER_OF_PATIENTS="3000"
NAME="${NUMBER_OF_PATIENTS}_GRAY"

echo $NAME

cd /dhc/home/tim.depping/GitHub/master_project
OUTPUT_DIR=/dhc/groups/mpws2022cl1/images/heart/png/$NAME
OUTPUT_CSV=/dhc/groups/mpws2022cl1/images/heart/png/$NAME.csv

python -u \
/dhc/home/tim.depping/GitHub/master_project/image_preprocessing.py \
--temp_dir ./.tmp_$NAME \
--output_dir $OUTPUT_DIR \
--number_of_patients $NUMBER_OF_PATIENTS \
--create_zip True \
--create_subdirs True \
--export_RGB_from_indices "" \
--indices ""

chgrp -R mpws2022cl1 $OUTPUT_DIR
chmod 770 $OUTPUT_DIR -R
chgrp mpws2022cl1 $OUTPUT_CSV
chmod 770 $OUTPUT_CSV

date