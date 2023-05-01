import os
import pathlib

from pydrake.all import RigidTransform, RollPitchYaw

from sim2sim.experiments import run_iiwa_manip

MANIPULAND_DIRECTIVE = (
    "../../models/iiwa_manip/iiwa_manip_pepper_manipuland_directive.yaml"
)
MANIPULAND_NAME = "pepper"
MANIPULAND_BASE_LINK_NAME = "pepper_base_link"
MANIPULANT_DEFAULT_POSE = RigidTransform(
    RollPitchYaw(0.0, 0.0, 0.0), [0.0, 0.45, 0.035]
)  # X_WManipuland


def run_iiwa_manip_pepper(
    logging_path: str,
    params: dict,
    sim_duration: float,
    timestep: float,
    logging_frequency_hz: float,
    hydroelastic_manipuland: bool,
):
    """
    Experiment entrypoint for the iiwa manip scenes.

    :param logging_path: The path to log the data to.
    :param params: The experiment yaml file dict.
    :param sim_duration: The simulation duration in seconds.
    :param timestep: The timestep to use in seconds.
    :param logging_frequency_hz: The dynamics logging frequency.
    :param hydroelastic_manipuland: Whether to use hydroelastic or point contact for the inner manipuland.
    """

    manipuland_directive = os.path.join(
        pathlib.Path(__file__).parent.resolve(), MANIPULAND_DIRECTIVE
    )

    run_iiwa_manip(
        logging_path,
        params,
        sim_duration,
        timestep,
        logging_frequency_hz,
        manipuland_directive,
        MANIPULANT_DEFAULT_POSE,
        MANIPULAND_BASE_LINK_NAME,
        MANIPULAND_NAME,
        hydroelastic_manipuland,
    )
