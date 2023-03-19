#!/bin/bash -eux
#SBATCH --job-name=plot
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.depping@student.hpi.de
#SBATCH --partition=hpcpu,vcpu,gpua100,gpupro,gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --export=ALL

OUTPUT_DIR=$1

eval "$(conda shell.bash hook)"
conda activate transfer_gwas

# Create mhat and qq plots
python -u lmm/create_plots.py \
--results_dir $OUTPUT_DIR/lmm/results \
--output_dir $OUTPUT_DIR/lmm/results/plots

# Change permissions for output dir
chgrp -R mpws2022cl1 $OUTPUT_DIR
chmod 770 $OUTPUT_DIR -R

date