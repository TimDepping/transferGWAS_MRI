#!/bin/bash -eux
#SBATCH --job-name=pca_extreme_values
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --time=4320
#SBATCH --partition=hpcpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --output=pca_extreme_values_%j.log # %j is job id
#SBATCH --export=ALL
 
# This script runs the python script that copies the images of indidivuals with extreme PC values and creates a csv file that contains their IIDs and the PC values.
# This job can be run after the LMM analysis.

date
hostname -f

python -u lmm/pca_extreme_values.py /dhc/groups/mpws2022cl1/output/50000_GRAY_50_minMaxScaling_ef_mc_2023_02_16_15_53_44_2023_03_06_11_29_52/resnet50_50_minMaxScaling_ef_mc_2023_02_16_15_53_44_L4.txt

echo "DONE"