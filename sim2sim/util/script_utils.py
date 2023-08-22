"""Utility functions used by the scripts."""

import os
from typing import List, Dict, Union
import time
import yaml

import numpy as np
from prettytable import PrettyTable
import wandb
import matplotlib.pyplot as plt

from sim2sim.experiments import (
    run_table_pid,
    run_iiwa_manip,
    run_floor_drop,
    run_random_force,
    run_planar_pushing,
)
from sim2sim.util import (
    trajectory_IoU,
    orientation_considered_final_displacement_error,
    orientation_considered_average_displacement_error,
    final_displacement_error_translation_only,
    average_displacement_error_translation_only,
    average_mean_contact_point_gradient_magnitude,
    average_generalized_contact_force_gradient_magnitude,
)

ENTRYPOINTS = {
    "table_pid": run_table_pid,
    "iiwa_manip": run_iiwa_manip,
    "floor_drop": run_floor_drop,
    "random_force": run_random_force,
    "planar_pushing": run_planar_pushing,
}


def create_evaluation_results_table(
    eval_data: List[Dict[str, float]],
    wandb_table: bool = False,
    sort_key: str = None,
) -> Union[PrettyTable, wandb.Table]:
    """
    :param wandb: Whether to create a wandb or PrettyTable table.
    """
    if sort_key is not None:
        eval_data.sort(key=lambda x: x[sort_key])

    column_names = list(eval_data[0].keys())
    data_rows = [[el[name] for name in column_names] for el in eval_data]

    if wandb_table:
        table = wandb.Table(data=data_rows, columns=column_names)
    else:
        table = PrettyTable(column_names)
        for row in data_rows:
            table.add_row(row)

    return table


def rank_based_on_final_errors(
    experiment_specifications: List[dict],
    logging_dir_path: str,
    quaternion_error_weight: float,
    translation_error_weight: float,
    angular_velocity_error_weight: float,
    translational_velocity_error_weight: float,
    log_wandb: bool = False,
    log_all_outer_htmls: bool = False,
) -> None:
    # TODO: Add options to parallelize this
    eval_data: List[Dict[str, float]] = []
    for i, experiment_specification in enumerate(experiment_specifications):
        name = experiment_specification["experiment_id"]
        print(f"\nEvaluating {name}:")

        logging_path = os.path.join(logging_dir_path, name)
        if not os.path.exists(logging_path):
            os.mkdir(logging_path)

        runner = ENTRYPOINTS[experiment_specification["script"]["name"]]
        runner(
            logging_path=logging_path,
            params=experiment_specification,
            **experiment_specification["script"]["args"],
        )

        time_logs_path = os.path.join(logging_path, "time_logs")
        outer_states = np.load(
            os.path.join(time_logs_path, "outer_manipuland_poses.npy")
        )
        inner_states = np.load(
            os.path.join(time_logs_path, "inner_manipuland_poses.npy")
        )
        # Take mean over manipulands
        final_state_error = np.mean(outer_states[:, -1] - inner_states[:, -1], axis=0)
        quaternion_error = np.linalg.norm(final_state_error[:4])
        translation_error = np.linalg.norm(final_state_error[4:7])
        angular_velocity_error = np.linalg.norm(final_state_error[7:10])
        translational_velocity_error = np.linalg.norm(final_state_error[10:])

        eval_data.append(
            {
                "name": name,
                "combined_error": quaternion_error_weight * quaternion_error
                + translation_error_weight * translation_error
                + angular_velocity_error_weight * angular_velocity_error
                + translational_velocity_error_weight * translational_velocity_error,
                "quaternion_error": quaternion_error,
                "translation_error": translation_error,
                "angular_velocity_error": angular_velocity_error,
                "translational_velocity_error": translational_velocity_error,
            }
        )

        print(
            f"Mesh: {name}, Translation err: {translation_error}, "
            + f"Quaternion err: {quaternion_error}, "
            + f"Translational velocity err: {translational_velocity_error}, "
            + f"Angular velocity err: {angular_velocity_error}"
        )

        if log_wandb:
            # Log simulation HTMLs to wandb
            inner_html = open(os.path.join(logging_path, "inner.html"))
            inner_name = f"{name}_inner" if log_all_outer_htmls else name
            wandb.log({inner_name: wandb.Html(inner_html, inject=False)})
            if log_all_outer_htmls or i == 0:
                outer_html = open(os.path.join(logging_path, "outer.html"))
                inner_name = f"{name}_outer" if log_all_outer_htmls else "outer"
                wandb.log({inner_name: wandb.Html(outer_html, inject=False)})

    if log_wandb:
        wandb_table = create_evaluation_results_table(eval_data, wandb_table=True)
        wandb.log({"evaluation_results": wandb_table})

    table = create_evaluation_results_table(eval_data, sort_key="combined_error")
    print("\n\nEvaluation results:\n")
    print(table)


def log_performance_time_plots(
    eval_data: List[Dict[str, float]],
    metric_keys: List[str],
    time_key: str,
    use_log_scale: bool = True,
):
    """
    Plot representation performance over simulation time and log with wandb.
    """
    scaler = (lambda x: np.log(x)) if use_log_scale else (lambda x: x)

    x_data = [scaler(errors[time_key]) for errors in eval_data]
    for metric in metric_keys:
        y_data = [scaler(errors[metric]) for errors in eval_data]

        fig_name = f"{metric}_over_{time_key}"
        fig, ax = plt.subplots()
        ax.scatter(x_data, y_data)
        ax.set_xlabel(f"log({time_key})" if use_log_scale else time_key)
        ax.set_ylabel(f"log({metric})" if use_log_scale else metric)

        for errors in eval_data:
            ax.annotate(
                errors["name"], (scaler(errors[time_key]), scaler(errors[metric]))
            )

        ax.autoscale()

        # NOTE: This requires matplotlib version < 3.5.0
        # (see https://github.com/plotly/plotly.py/issues/3624)
        wandb.log({fig_name: fig})
        plt.close(fig)


def process_contact_points_arr(arr: np.ndarray) -> np.ndarray:
    max_count = 0
    for el in arr:
        max_count = max(max_count, len(el))

    return np.array(
        [
            np.concatenate([el, np.zeros((max_count - len(el), 3))], axis=0)
            if len(el) > 0
            else np.zeros((max_count, 3))
            for el in arr
        ]
    )


def rank_based_on_metrics(
    experiment_specifications: List[dict],
    logging_dir_path: str,
    num_trajectory_iou_samples: int,
    log_wandb: bool = False,
    log_all_outer_htmls: bool = False,
    additional_metric_keys: List[str] = [],
) -> None:
    def create_table_dict(
        name: str,
        trajectory_IoU_margin_01: float,
        trajectory_IoU_margin_1: float,
        orientation_considered_final_error: float,
        orientation_considered_average_error: float,
        final_translation_error: float,
        average_translation_error: float,
        average_mean_contact_point_gradient_magnitude: float,
        average_generalized_contact_force_gradient_magnitude: float,
        simulation_time: float,
        simulation_time_ratio: float,  # inner_time / outer_time
    ) -> Dict[str, Union[str, float]]:
        return {
            "name": name,
            "trajectory_IoU_margin_01": trajectory_IoU_margin_01,
            "trajectory_IoU_margin_1": trajectory_IoU_margin_1,
            "orientation_considered_final_error": orientation_considered_final_error,
            "orientation_considered_average_error": orientation_considered_average_error,
            "final_translation_error": final_translation_error,
            "average_translation_error": average_translation_error,
            "average_mean_contact_point_gradient_magnitude": average_mean_contact_point_gradient_magnitude,
            "average_generalized_contact_force_gradient_magnitude": average_generalized_contact_force_gradient_magnitude,
            "simulation_time": simulation_time,
            "simulation_time_ratio": simulation_time_ratio,
        }

    # TODO: Add options to parallelize this
    eval_data: List[Dict[str, float]] = []
    for i, experiment_specification in enumerate(experiment_specifications):
        name = experiment_specification["experiment_id"]
        print(f"\nEvaluating {name}:")

        logging_path = os.path.join(logging_dir_path, name)
        if not os.path.exists(logging_path):
            os.mkdir(logging_path)

        runner = ENTRYPOINTS[experiment_specification["script"]["name"]]
        runner(
            logging_path=logging_path,
            params=experiment_specification,
            **experiment_specification["script"]["args"],
        )

        time_logs_path = os.path.join(logging_path, "time_logs")
        # States
        outer_states = np.load(
            os.path.join(time_logs_path, "outer_manipuland_poses.npy")
        )
        inner_states = np.load(
            os.path.join(time_logs_path, "inner_manipuland_poses.npy")
        )
        # Generalized contact forces
        inner_generalized_contact_force_torques = np.load(
            os.path.join(time_logs_path, "inner_manipuland_contact_forces.npy")
        )
        inner_generalized_contact_forces = inner_generalized_contact_force_torques[
            :, :, 3:
        ]  # Shape (M, N, 3)
        # Contact points
        hydroelastic_contact_points_raw = np.load(
            os.path.join(
                time_logs_path, "inner_hydroelastic_contact_result_centroids.npy"
            ),
            allow_pickle=True,
        )
        point_contact_points_raw = np.load(
            os.path.join(
                time_logs_path, "inner_point_contact_result_contact_points.npy"
            ),
            allow_pickle=True,
        )
        contact_points_raw = (
            hydroelastic_contact_points_raw
            if any(hydroelastic_contact_points_raw)
            else point_contact_points_raw
        )
        inner_contact_points = process_contact_points_arr(contact_points_raw)

        meta_data_path = os.path.join(logging_path, "meta_data.yaml")
        meta_data = yaml.safe_load(open(meta_data_path))

        start_time = time.time()
        # Compute metric for each manipuland separately
        trajectory_IoU_margins_01 = [
            trajectory_IoU(
                outer,
                inner,
                margin=0.1,
                num_samples=num_trajectory_iou_samples,
            )
            for outer, inner in zip(outer_states, inner_states)
        ]  # Shape (M,)
        trajectory_IoU_margins_1 = [
            trajectory_IoU(
                outer,
                inner,
                margin=1.0,
                num_samples=num_trajectory_iou_samples,
            )
            for outer, inner in zip(outer_states, inner_states)
        ]  # Shape (M,)
        orientation_considered_final_errors = [
            orientation_considered_final_displacement_error(outer, inner)
            for outer, inner in zip(outer_states, inner_states)
        ]  # Shape (M,)
        orientation_considered_average_errors = [
            orientation_considered_average_displacement_error(outer, inner)
            for outer, inner in zip(outer_states, inner_states)
        ]  # Shape (M,)
        final_displacement_errors_translation_only = [
            final_displacement_error_translation_only(outer, inner)
            for outer, inner in zip(outer_states, inner_states)
        ]  # Shape (M,)
        average_displacement_errors_translation_only = [
            average_displacement_error_translation_only(outer, inner)
            for outer, inner in zip(outer_states, inner_states)
        ]  # Shape (M,)
        # TODO: Filter contact points based on manipuland and only consider the ones
        # belonging to the manipuland whose states are used
        average_mean_contact_point_gradient_magnitudes = [
            average_mean_contact_point_gradient_magnitude(inner_contact_points, states)
            for states in inner_states
        ]  # Shape (M,)
        average_generalized_contact_force_gradient_magnitudes = [
            average_generalized_contact_force_gradient_magnitude(forces)
            for forces in inner_generalized_contact_forces
        ]  # Shape (M,)
        # Average metrics over manipulands
        errors = create_table_dict(
            name=name,
            trajectory_IoU_margin_01=np.mean(trajectory_IoU_margins_01),
            trajectory_IoU_margin_1=np.mean(trajectory_IoU_margins_1),
            orientation_considered_final_error=np.mean(
                orientation_considered_final_errors
            ),
            orientation_considered_average_error=np.mean(
                orientation_considered_average_errors
            ),
            final_translation_error=np.mean(final_displacement_errors_translation_only),
            average_translation_error=np.mean(
                average_displacement_errors_translation_only
            ),
            average_mean_contact_point_gradient_magnitude=np.mean(
                average_mean_contact_point_gradient_magnitudes
            ),
            average_generalized_contact_force_gradient_magnitude=np.mean(
                average_generalized_contact_force_gradient_magnitudes
            ),
            simulation_time=meta_data["time_taken_to_simulate_inner_s"],
            simulation_time_ratio=meta_data["time_taken_to_simulate_inner_s"]
            / meta_data["time_taken_to_simulate_outer_s"],
        )
        for additional_metric in additional_metric_keys:
            errors[additional_metric] = (
                experiment_specification[additional_metric]
                if additional_metric in experiment_specification
                else np.nan
            )
        eval_data.append(errors)

        print(f"Computing errors took {time.time()-start_time} seconds.")
        print(f"Mesh: {name}, Errors:\n{errors}")

        if log_wandb:
            # Log simulation HTMLs to wandb
            inner_html = open(os.path.join(logging_path, "inner.html"))
            inner_name = f"{name}_inner" if log_all_outer_htmls else name
            wandb.log({inner_name: wandb.Html(inner_html, inject=False)})
            if log_all_outer_htmls or i == 0:
                outer_html = open(os.path.join(logging_path, "outer.html"))
                inner_name = f"{name}_outer" if log_all_outer_htmls else "outer"
                wandb.log({inner_name: wandb.Html(outer_html, inject=False)})

    if log_wandb:
        wandb_table = create_evaluation_results_table(
            eval_data, wandb_table=True, sort_key="orientation_considered_average_error"
        )
        wandb.log({"evaluation_results": wandb_table})

        log_performance_time_plots(
            eval_data,
            metric_keys=[
                "trajectory_IoU_margin_01",
                "trajectory_IoU_margin_1",
                "orientation_considered_final_error",
                "orientation_considered_average_error",
                "final_translation_error",
                "average_translation_error",
                "average_mean_contact_point_gradient_magnitude",
                "average_generalized_contact_force_gradient_magnitude",
            ],
            time_key="simulation_time_ratio",
        )

    table = create_evaluation_results_table(
        eval_data, sort_key="orientation_considered_average_error"
    )
    print("\n\nEvaluation results:\n")
    print(table)
