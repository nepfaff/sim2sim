# sim2sim
A simulation-based evaluation system for real2sim approaches

A high-level system diagram is shown below:
![system_overview_diagram](system_overview_diagram.png)

## Installation

Clone the repo and execute the following commands from the repository's root.

Install `git lfs`:

```bash
git lfs install
git lfs pull
```

Install the `sim2sim` package in development mode:

```bash
pip install -e .
```

Install `pre-commit` for automatic black formatting:
```bash
pre-commit install
```

Install the [learning_real2sim](https://github.com/liruiw/learning_real2sim) repo in development mode based on the
instructions in its `README`. Note that this repo is only needed for a small subset of experiments.

## Experiment Types

This repo provides two different experiment types. The first one is the sim2sim pipeline experiment as shown in the
system overview diagram. The second one is for comparing two different sim2sim pipelines by running one of them as the
'outer' and one of them as the 'inner'. The main difference here is that the sim2sim pipeline is duplicated and one of
them is run instead of the hand-crafted 'outer'/ real-world manipuland. This is particularly useful for evaluating how
two different sim2sim pipelines differ from each other (e.g. by using the contact force visualizer).

**NOTE:** The sim2im pipeline comparison experiment type is currently only supported by the `sphere_pushing` and `table_pid` experiments.

The sim2sim pipeline comparison experiment type is shown in the image below:
![pipeline_comparison_diagram](pipeline_comparison_diagram.png)

### Experiment description files

An experiment description file deterministically specifies an experiment. New experiments can easily be constructed by
mixing components and their parameters in an experiment description file.

An example of a sim2sim pipeline experiment can be found in `experiments/sphere_pushing/sphere_pushing_coacd.yaml` and
an example of a sim2sim pipeline comparison experiment can be found in
`experiments/sphere_pushing/sphere_pushing_coacd_vs_dualsdf_sphere.yaml`. These two look very similar apart from the
`is_pipeline_comparison` parameter and that there is an `inner` and `outer` version for most components in the pipeline
comparison experiment.

## Running an experiment

Replace `experiments/table_pid/table_pid_simple.yaml` in the command below with your experiment description file.
The experiment description file deterministically specifies an experiment.

```bash
python scripts/run_experiment.py --experiment_description experiments/sphere_pushing/sphere_pushing_coacd.yaml
```

## Contact Force Visualizer

The visualizer can be used to visualize both `outer` and `inner` manipulands and their contact forces over time.

Example usage:
```bash
python3 scripts/visualize_contact_forces.py --data logs/sphere_pushing_coacd/ --separation_distance 0.2
```

Arrow colors:
- ![#0000FF](https://placehold.co/15x15/0000FF/0000FF.png) Outer generalized contact force (force component)
- ![#9999FF](https://placehold.co/15x15/9999FF/9999FF.png) Outer generalized contact force (torque component)
- ![#FF0080](https://placehold.co/15x15/FF0080/FF0080.png) Inner generalized contact force (force component)
- ![#FF99CC](https://placehold.co/15x15/FF99CC/FF99CC.png) Inner generalized contact force (torque component)
- ![#00FF00](https://placehold.co/15x15/00FF00/00FF00.png) Outer point/ hydroelastic contact force
- ![#00FFFF](https://placehold.co/15x15/00FFFF/00FFFF.png) Outer hydroelastic contact torque
- ![#FF0000](https://placehold.co/15x15/FF0000/FF0000.png) Inner point/ hydroelastic contact force
- ![#FF8000](https://placehold.co/15x15/FF8000/FF8000.png) Inner hydroelastic contact torque

It is possible to step through time using the left and right arrow keys. The "toggle" buttons in meshcat can be used to
show and hide items by default when stepping through time.
See `scripts/visualize_contact_forces.py` for all available arguments.

## Evaluating real2sim approaches quantitatively

### Evaluation based on errors at final timestep

The following script can be used for ranking different real2sim approaches based on translation, rotation, and velocity
errors at the final timestep.

```bash
python scripts/rank_real2sim_approaches_final_errors.py --experiment_descriptions experiments/floor_drop/
```

where `experiments/floor_drop/` is a folder containing experiment description files.

The intended setup are experiment description files that describe the same experiment but with different real2sim
approaches. For example, they may have different real2sim pipeline components or different component parameters.
The experiment description files are ranked based on a weighted combination of the individual errors. The weights are
hardcoded in `scripts/rank_real2sim_approaches_final_errors.py`.

### Evaluating a primitive representation collection based on errors at the final timestep

The following script can be used for ranking representations in a primitive representation based on translation, rotation,
and velocity errors at the final timestep.

```bash
python scripts/evaluate_primitive_representation_collection.py --path representation_collection/ --experiment_description experiments/sphere_pushing/sphere_pushing_mustard_ring_coacd_vs_spheres.yaml
```

where `representation_collection/` is a folder with the following structure:
- representation_collection/
    - representation1/
        - primitive_info.pkl
    - representation2/
        - primitive_info.pkl
    - ...
    - representationN/
        - primitive_info.pkl
    - physical_properties.yaml

`physical_properties.yaml` must contain a `mass` and a `inertia` field.

The experiment description specifies the experiment and must contain the following components:
- `IdentityPrimitiveMeshProcessor` for `inner_mesh_processor`
- `GTPhysicalPropertyEstimator` for `outer_physical_property_estimator`
- `GTPhysicalPropertyEstimator` for `inner_physical_property_estimator`

**NOTE:** Make sure that `inner_env` uses the desired contact model for simulating the representation collection (`point` or `hydroelastic_with_fallback`).

## Generating a mesh dynamic distance dataset

**NOTE:** Currently only the `random_force` experiment is supported.

1. Specify the desired experiment parameters in `experiments/random_force/random_force_metaball.yaml`.
2. Generate the data using one of the following commands (with your arguments):
    ```bash
    python scripts/collect_random_force_data.py --experiment_description experiments/random_force/random_force_gmm.yaml --logging_path logs/metric_learning_data --num_runs_per_perturbation 10 --num_perturbations 1000
    ```
    ```bash
    python3 scripts/collect_sphere_pushing_data.py --experiment_description experiments/sphere_pushing/sphere_pushing_gmm.yaml --logging_path logs/metric_learning_data --num_runs_per_perturbation 10 --num_perturbations 1000
    ```
3. Postprocess the data:
    ```bash
    python scripts/postprocess_metric_learning_dataset.py --data_path logs/metric_learning_data
    ```
    
### Generating a metric for a specific representation

1. Define the representation/ mesh processor in a `random_force.yaml` file.
2. Generate the data using the following command (with your arguments):
    ```bash
    python scripts/collect_representation_specific_random_force_data.py --experiment_description experiments/random_force/random_force.yaml --logging_path logs/mean_random_force --num_runs 50
    ```
3. Postprocess the data:
    ```bash
    python scripts/postprocess_metric_learning_dataset.py --is_representation_specific --data_path logs/mean_random_force
    ```
    
## Generating simulator timing scale data with number of ellipsoids

1. Comment out all meshcat specific stuff in `simulation/random_force_simulator.py` (Drake crashes if more than 100
meshcats are spawned and it is not possible to clean them up without terminating the top-level script).
2. Collect the data:
    ```bash
    python scripts/collect_simulation_complexity_data.py --logging_path logs/simulation_complexity --experiment_description experiments/random_force/random_force_gmm.yaml
    ```
3. Create the plot in `logs/simulation_complexity':
    ```bash
    python scripts/postprocess_simulation_complexity_data.py --data_path logs/simulation_complexity/
    ```
    
## Data format

The camera extrinsics are homogenous `world2cam` transformation matrices with `OpenCV convention`.

## Requirements for using mesh decomposition

1. Install [pointnet-pytorch](https://github.com/liruiw/Pointnet2_PyTorch).
2. Install [v-hacd](https://github.com/mikedh/trimesh/blob/30a423b884903905aba82408255f02dec0b33175/docker/builds/vhacd.bash)
in trimesh by running the script.
3. Install [CoACD](https://github.com/liruiw/CoACD). Copy the binaries to the system bin with the name `coacd`.

## Error: "Meshes does not have textures"

This requires modifying the pytorch3d python files as suggested [here](https://github.com/facebookresearch/pytorch3d/issues/333#issuecomment-678129430) (remember to do it for all the shaders, the line numbers are no longer accurate).
