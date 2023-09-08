#!/bin/bash

# Script for training/ evaluation in non-interactive sessions on supercloud
# Submit via `LLsub supercloud.sh -s 20 -g volta:1` and show status via `LLstat`
# Logs are written to `supercloud.sh.log-{job_id}`

source /etc/profile

module load anaconda/2022b
module load gurobi/gurobi-1000

. sim2sim_env/bin/activate

bash scripts/experiments.sh
