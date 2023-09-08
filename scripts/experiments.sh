#!/bin/bash

RUN_NUMBER=${1}

bash experiments/icra/planar_pushing/experiments.sh ${RUN_NUMBER}
bash experiments/icra/table_pid/experiments.sh ${RUN_NUMBER}
bash experiments/icra/interaction_pushing/experiments.sh ${RUN_NUMBER}
