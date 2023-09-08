#!/bin/bash

# Script for training/ evaluation in non-interactive sessions on supercloud
# Submit via `LLsub scripts/supercloud.sh -s 48` and show status via `LLstat`
# If max allowed, use `LLsub scripts/supercloud.sh -s 40 -g volta:2`
# Logs are written to `supercloud.sh.log-{job_id}`

source /etc/profile

module load anaconda/2022b
module load gurobi/gurobi-1000

. sim2sim_env/bin/activate

# Dummy run to compile everything
python scripts/run_experiment.py \
--experiment_specification experiments/planar_pushing/box/mustard_raw_tsdf_vs_spheres_equation_error.yaml

# Actual experiments
bash scripts/experiments.sh
