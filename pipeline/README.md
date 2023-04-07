# Pipeline

`pipeline.sh` contains the complete pipeline to
1. run the feature condensation
2. run the LMM analysis
3. and to create plots to visualize the results of the LMM analysis.

For each step of the pipeline a corresponding slurm job will be started on the HPC Cluster.
The slurm jobs call the python and bash scripts of this repository.
For this reason, `pipeline.sh` should be run from the root of the repository and the Conda environment needs to be activated.