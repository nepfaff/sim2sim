# sim2sim
A simulation-based evaluation system for real2sim approaches

A high-level system diagram is shown below:
![system_overview_diagram](system_overview_diagram.png)

## Installation

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
