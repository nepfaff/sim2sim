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

## Running an experiment

Replace `experiments/table_pid/table_pid_simple.yaml` in the command below with your experiment description file.
The experiment description file deterministically specifies an experiment.

```bash
python scripts/run_experiment.py --experiment_description experiments/table_pid/table_pid_simple.yaml
```

## Contact Force Visualizer

The visualizer can be used to visualize both `outer` and `inner` manipulands and their contact forces over time.

Example usage:
```bash
python3 scripts/visualize_contact_forces.py --data logs/sphere_pushing_quadric_decimation/
```

It is possible to step through time using the left and right arrow keys.
See `scripts/visualize_contact_forces.py` for all available arguments.

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

1. Comment out all meshcat specific stuff in `simulation/random_force_simulator.py` (Drake crashes if more than 100 meshcats are spawned and it is not possible to clean them up without terminating the top-level script).
2. Collect the data:
    ```bash
    python scripts/collect_simulation_complexity_data.py --logging_path logs/simulation_complexity --experiment_description experiments/random_force/random_force_gmm.yaml
    ```
3. Create the plot in `logs/simulation_complexity':
    ```bash
    python scripts/postprocess_simulation_complexity_data.py --data_path logs/simulation_complexity/
    ```

## Requirements for using mesh decomposition

1. Install [pointnet-pytorch](https://github.com/liruiw/Pointnet2_PyTorch).
2. Install [v-hacd](https://github.com/mikedh/trimesh/blob/30a423b884903905aba82408255f02dec0b33175/docker/builds/vhacd.bash) in trimesh by running the script.
3. Install [CoACD](https://github.com/liruiw/CoACD). Copy the binaries to the system bin with the name `coacd`.

## Error: "Meshes does not have textures"

This requires modifying the pytorch3d python files as suggested [here](https://github.com/facebookresearch/pytorch3d/issues/333#issuecomment-678129430) (remember to do it for all the shaders, the line numbers are no longer accurate).
