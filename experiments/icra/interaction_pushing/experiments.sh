#!/bin/bash

RUN_NUMBER=${1}
PREFIX="icra"

# Anvil onto dumbell
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_dumbell_sept/",
    "~/representation_collections/representation_collection_anvil_sept/"
    ]' \
--experiment_description experiments/icra/interaction_pushing/anvil_onto_dumbell.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/dumbell",
    "~/coacd_vhacd_mesh_pieces/anvil"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_interaction_pushing_anvil_onto_dumbell_${RUN_NUMBER}

# Mustard onto tomato
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_mustard_ring3_nerfstudio_sept/",
    "~/representation_collections/representation_collection_tomato_can_nerfstudio_sept/"
    ]' \
--experiment_description experiments/icra/interaction_pushing/mustard_onto_tomato.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/mustard",
    "~/coacd_vhacd_mesh_pieces/tomato"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_interaction_pushing_mustard_onto_tomato_${RUN_NUMBER}

# Spam onto anvil
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_spam_sept/",
    "~/representation_collections/representation_collection_anvil_sept/"
    ]' \
--experiment_description experiments/icra/interaction_pushing/spam_onto_anvil.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/spam",
    "~/coacd_vhacd_mesh_pieces/anvil"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_interaction_pushing_spam_onto_anvil_${RUN_NUMBER}

# Spam onto mustard
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_spam_sept/",
    "~/representation_collections/representation_collection_mustard_ring3_nerfstudio_sept/"
    ]' \
--experiment_description experiments/icra/interaction_pushing/spam_onto_mustard.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/spam",
    "~/coacd_vhacd_mesh_pieces/mustard"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_interaction_pushing_spam_onto_mustard_${RUN_NUMBER}

# Tomato onto hammer
python scripts/evaluate_primitive_representation_collection.py \
--paths '[
    "~/representation_collections/representation_collection_tomato_can_nerfstudio_sept/",
    "~/representation_collections/representation_collection_hammer_sept/"
    ]' \
--experiment_description experiments/icra/interaction_pushing/tomato_onto_hammer.yaml \
--eval_contact_model \
--additional_collision_geometries_mesh_pieces_paths '[
    "~/coacd_vhacd_mesh_pieces/tomato",
    "~/coacd_vhacd_mesh_pieces/hammer"
    ]' \
--num_trajectory_iou_samples 0 \
--include_gt \
--wandb_mode "offline" \
--wandb_name ${PREFIX}_interaction_pushing_tomato_onto_hammer_${RUN_NUMBER}
