#!/bin/bas
RUN_NUMBER=${1}

## NOTE: This is a convenience script that should NOT be commited

# Tomato can table PID, equation error
python scripts/evaluate_primitive_representation_collection.py \
--path ~/robot_locomotion/DualSDF/datasets/representation_collection_tomato_can_nerfstudio/ \
--eval_contact_model \
--experiment_description experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_spheres_equation_error.yaml \
--additional_experiment_descriptions '["experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_quadric_decimation_equation_error.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_coacd_equation_error.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_coacd_point_equation_error.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_quadric_decimation_point_equation_error.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_quadric_decimation1k_equation_error.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_quadric_decimation2k_equation_error.yaml"]' \
--num_trajectory_iou_samples 0 \
--wandb_name table_pid_equation_error_tomato_can_${RUN_NUMBER}

# Mustard bottle table PID, equation error
python scripts/evaluate_primitive_representation_collection.py \
--path ~/robot_locomotion/DualSDF/datasets/representation_collection_mustard_ring3_nerfstudio/ \
--eval_contact_model \
--experiment_description experiments/table_pid/table_pid_mustard_raw_tsdf_vs_spheres_equation_error.yaml \
--additional_experiment_descriptions '["experiments/table_pid/table_pid_mustard_raw_tsdf_vs_quadric_decimation_equation_error.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_coacd_equation_error.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_coacd_point_equation_error.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_quadric_decimation_point_equation_error.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_quadric_decimation1k_equation_error.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_quadric_decimation2k_equation_error.yaml"]' \
--num_trajectory_iou_samples 0 \
--wandb_name table_pid_equation_error_mustard_${RUN_NUMBER}

# Tomato can table PID, simulation error
python scripts/evaluate_primitive_representation_collection.py \
--path ~/robot_locomotion/DualSDF/datasets/representation_collection_tomato_can_nerfstudio/ \
--eval_contact_model \
--experiment_description experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_spheres.yaml \
--additional_experiment_descriptions '["experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_quadric_decimation.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_coacd.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_coacd_point.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_quadric_decimation_point.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_quadric_decimation1k.yaml", "experiments/table_pid/table_pid_tomato_can_raw_tsdf_vs_quadric_decimation2k.yaml"]' \
--num_trajectory_iou_samples 0 \
--wandb_name table_pid_simulation_error_tomato_can_${RUN_NUMBER}

# Mustard bottle table PID, simulation error
python scripts/evaluate_primitive_representation_collection.py \
--path ~/robot_locomotion/DualSDF/datasets/representation_collection_mustard_ring3_nerfstudio/ \
--eval_contact_model \
--experiment_description experiments/table_pid/table_pid_mustard_raw_tsdf_vs_spheres.yaml \
--additional_experiment_descriptions '["experiments/table_pid/table_pid_mustard_raw_tsdf_vs_quadric_decimation.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_coacd.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_coacd_point.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_quadric_decimation_point.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_quadric_decimation1k.yaml", "experiments/table_pid/table_pid_mustard_raw_tsdf_vs_quadric_decimation2k.yaml"]' \
--num_trajectory_iou_samples 0 \
--wandb_name table_pid_simulation_error_mustard_${RUN_NUMBER}

# Tomato can sphere pushing, equation error
python scripts/evaluate_primitive_representation_collection.py \
--path ~/robot_locomotion/DualSDF/datasets/representation_collection_tomato_can_nerfstudio/ \
--eval_contact_model \
--experiment_description experiments/sphere_pushing/tomato_can_raw_tsdf_vs_spheres_equation_error.yaml \
--additional_experiment_descriptions '["experiments/sphere_pushing/tomato_can_raw_tsdf_vs_quadric_decimation_equation_error.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_coacd_equation_error.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_coacd_point_equation_error.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_quadric_decimation_point_equation_error.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_quadric_decimation1k_equation_error.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_quadric_decimation2k_equation_error.yaml"]' \
--num_trajectory_iou_samples 0 \
--wandb_name sphere_pushing_equation_error_tomato_can_${RUN_NUMBER}

# Mustard bottle sphere pushing, equation error
python scripts/evaluate_primitive_representation_collection.py \
--path ~/robot_locomotion/DualSDF/datasets/representation_collection_mustard_ring3_nerfstudio/ \
--eval_contact_model \
--experiment_description experiments/sphere_pushing/mustard_raw_tsdf_vs_spheres_equation_error.yaml \
--additional_experiment_descriptions '["experiments/sphere_pushing/mustard_raw_tsdf_vs_quadric_decimation_equation_error.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_coacd_equation_error.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_coacd_point_equation_error.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_quadric_decimation_point_equation_error.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_quadric_decimation1k_equation_error.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_quadric_decimation2k_equation_error.yaml"]' \
--num_trajectory_iou_samples 0 \
--wandb_name sphere_pushing_equation_error_mustard_${RUN_NUMBER}

# Tomato can sphere pushing, simulation error
python scripts/evaluate_primitive_representation_collection.py \
--path ~/robot_locomotion/DualSDF/datasets/representation_collection_tomato_can_nerfstudio/ \
--eval_contact_model \
--experiment_description experiments/sphere_pushing/tomato_can_raw_tsdf_vs_spheres.yaml \
--additional_experiment_descriptions '["experiments/sphere_pushing/tomato_can_raw_tsdf_vs_quadric_decimation.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_coacd.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_coacd_point.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_quadric_decimation_point.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_quadric_decimation1k.yaml", "experiments/sphere_pushing/tomato_can_raw_tsdf_vs_quadric_decimation2k.yaml"]' \
--num_trajectory_iou_samples 0 \
--wandb_name sphere_pushing_simulation_error_tomato_can_${RUN_NUMBER}

# Mustard bottle sphere pushing, simulation error
python scripts/evaluate_primitive_representation_collection.py \
--path ~/robot_locomotion/DualSDF/datasets/representation_collection_mustard_ring3_nerfstudio/ \
--eval_contact_model \
--experiment_description experiments/sphere_pushing/mustard_raw_tsdf_vs_spheres.yaml \
--additional_experiment_descriptions '["experiments/sphere_pushing/mustard_raw_tsdf_vs_quadric_decimation.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_coacd.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_coacd_point.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_quadric_decimation_point.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_quadric_decimation1k.yaml", "experiments/sphere_pushing/mustard_raw_tsdf_vs_quadric_decimation2k.yaml"]' \
--num_trajectory_iou_samples 0 \
--wandb_name sphere_pushing_simulation_error_mustard_${RUN_NUMBER}
