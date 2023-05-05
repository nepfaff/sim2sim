"""Utility functions used by the scripts."""

import os
from typing import List, Dict, Union
import time
import yaml

import numpy as np
from prettytable import PrettyTable
import wandb

from sim2sim.experiments import (
    run_table_pid,
    run_iiwa_manip_tomato_soup_can,
    run_iiwa_manip_pepper,
    run_floor_drop,
    run_random_force,
    run_sphere_pushing,
)
from sim2sim.util import (
    trajectory_IoU,
    orientation_considered_final_displacement_error,
    orientation_considered_average_displacement_error,
    final_displacement_error_translation_only,
    average_displacement_error_translation_only,
)

ENTRYPOINTS = {
    "table_pid": run_table_pid,
    "iiwa_manip_tomato_soup_can": run_iiwa_manip_tomato_soup_can,
    "iiwa_manip_pepper": run_iiwa_manip_pepper,
    "floor_drop": run_floor_drop,
    "random_force": run_random_force,
    "sphere_pushing": run_sphere_pushing,
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
        outer_states = np.loadtxt(
            os.path.join(time_logs_path, "outer_manipuland_poses.txt")
        )
        inner_states = np.loadtxt(
            os.path.join(time_logs_path, "inner_manipuland_poses.txt")
        )
        final_state_error = outer_states[-1] - inner_states[-1]
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
            f"Mesh: {name}, Translation err: {translation_error}, Quaternion err: {quaternion_error}, "
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
        outer_states = np.loadtxt(
            os.path.join(time_logs_path, "outer_manipuland_poses.txt")
        )
        inner_states = np.loadtxt(
            os.path.join(time_logs_path, "inner_manipuland_poses.txt")
        )

        meta_data_path = os.path.join(logging_path, "meta_data.yaml")
        meta_data = yaml.safe_load(open(meta_data_path))

        start_time = time.time()
        errors = create_table_dict(
            name=name,
            trajectory_IoU_margin_01=trajectory_IoU(
                outer_states,
                inner_states,
                margin=0.1,
                num_samples=num_trajectory_iou_samples,
            ),
            trajectory_IoU_margin_1=trajectory_IoU(
                outer_states,
                inner_states,
                margin=1.0,
                num_samples=num_trajectory_iou_samples,
            ),
            orientation_considered_final_error=orientation_considered_final_displacement_error(
                outer_states, inner_states
            ),
            orientation_considered_average_error=orientation_considered_average_displacement_error(
                outer_states, inner_states
            ),
            final_translation_error=final_displacement_error_translation_only(
                outer_states, inner_states
            ),
            average_translation_error=average_displacement_error_translation_only(
                outer_states, inner_states
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
            eval_data, wandb_table=True, sort_key="orientation_considered_final_error"
        )
        wandb.log({"evaluation_results": wandb_table})

    table = create_evaluation_results_table(
        eval_data, sort_key="orientation_considered_final_error"
    )
    print("\n\nEvaluation results:\n")
    print(table)
