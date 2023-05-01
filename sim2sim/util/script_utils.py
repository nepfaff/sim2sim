"""Utility functions used by the scripts."""

import os
from typing import List, Dict, Union

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

ENTRYPOINTS = {
    "table_pid": run_table_pid,
    "iiwa_manip_tomato_soup_can": run_iiwa_manip_tomato_soup_can,
    "iiwa_manip_pepper": run_iiwa_manip_pepper,
    "floor_drop": run_floor_drop,
    "random_force": run_random_force,
    "sphere_pushing": run_sphere_pushing,
}


def create_evaluation_results_table(
    eval_data: List[Dict[str, float]], wandb_table: bool = False
) -> Union[PrettyTable, wandb.Table]:
    """
    :param wandb: Whether to create a wandb or PrettyTable table.
    """
    eval_data.sort(key=lambda x: x["combined_error"])

    column_names = [
        "Mesh name",
        "Combined Err",
        "Translation Err",
        "Quaternion Err",
        "Translational Velocity Err",
        "Angular Velocity Err",
    ]
    data_rows = [
        [
            el["name"],
            el["combined_error"],
            el["translation_error"],
            el["quaternion_error"],
            el["translational_velocity_error"],
            el["angular_velocity_error"],
        ]
        for el in eval_data
    ]

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

    table = create_evaluation_results_table(eval_data)
    print("\n\nEvaluation results:\n")
    print(table)
