#!/bin/bash

RUN_NUMBER=${1}
PREFIX="icra"

# Mustard
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_mustard_ring3_nerfstudio_sept/"
    ]' \
--experiment_description experiments/icra/table_pid/mustard_raw_tsdf_vs_spheres_equation_error.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/mustard"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_table_pid_mustard_${RUN_NUMBER}

# Tomato
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_tomato_can_nerfstudio_sept/"
    ]' \
--experiment_description experiments/icra/table_pid/tomato_can_raw_tsdf_vs_spheres_equation_error.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/tomato"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_table_pid_tomato_${RUN_NUMBER}

# Anvil
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_anvil_sept/"
    ]' \
--experiment_description experiments/icra/table_pid/anvil_raw_tsdf_vs_spheres_equation_error.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/anvil"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_table_pid_anvil_${RUN_NUMBER}

# Dumbell
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_dumbell_sept/"
    ]' \
--experiment_description experiments/icra/table_pid/dumbell_raw_tsdf_vs_spheres_equation_error.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/dumbell"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_table_pid_dumbell_${RUN_NUMBER}

# Hammer
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_hammer_sept/"
    ]' \
--experiment_description experiments/icra/table_pid/hammer_raw_tsdf_vs_spheres_equation_error.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/hammer"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_table_pid_hammer_${RUN_NUMBER}

# Spam
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_spam_sept/"
    ]' \
--experiment_description experiments/icra/table_pid/spam_raw_tsdf_vs_spheres_equation_error.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/spam"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_table_pid_spam_${RUN_NUMBER}
