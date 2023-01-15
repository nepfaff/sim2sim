# sim2sim
A simulation-based evaluation system for real2sim approaches

A high-level system diagram is shown below:
![system_overview_diagram](system_overview_diagram.png)

## Installation

Execute the following commands from the repository's root.

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

## Running an experiment

Replace `experiments/table_pid/table_pid_simple.yaml` in the command below with your experiment description file.
The experiment description file deterministically specifies an experiment.

```bash
python scripts/run_experiment.py --experiment_description experiments/table_pid/table_pid_simple.yaml
```

## Generating a mesh dynamic distance dataset

**NOTE:** Currently only the `random_force` experiment is supported.

1. Specify the desired experiment parameters in `experiments/random_force/random_force.yaml`.
2. Generate the data using the following command (with your arguments):
    ```bash
    python scripts/collect_random_force_data.py --experiment_description experiments/random_force/random_force.yaml --logging_path      logs/metric_learning_data --num_runs_per_perturbation 10 --num_perturbations 1000
    ```
3. Postprocess the data:
    ```bash
    python scripts/construct_metric_learning_dataset.py --data_path logs/metric_learning_data
    ```

## Requirements for using mesh decomposition

1. Install [pointnet-pytorch](https://github.com/liruiw/Pointnet2_PyTorch).
2. Install [v-hacd](https://github.com/mikedh/trimesh/blob/30a423b884903905aba82408255f02dec0b33175/docker/builds/vhacd.bash) in trimesh by running the script.
3. Install [CoACD](https://github.com/liruiw/CoACD). Copy the binaries to the system bin with the name `coacd`.
