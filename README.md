# An Analysis of the Coordination Gap between Joint and Modular Learning for Job Shop Scheduling with Transportation Resources

This repository implements the Framework for the Job Shop Scheduling Problem with Transportation resources (JSSPT), as presented in our research. 

## Overview
- `src/jssp_core`: A framework-agnostic scheduling environment for the JSSPT, observation/reward registries, and heuristic baselines.
- `src/jssp_agv`: Training files (train_gnn_aec.py, train_gnn_jssp.py,train_gnn_agv.py), experiment configs and executable files, dispatcher for training, component setup, and GNN models. 
- `src/jssp_gnn`: Basic components for the training. The GNN models are defined in the jssp_agv folder.
- `config`: Config files for the training
- `Release:json_export`: Data of the experiments in JSON.
- `src/jssp_agv/paper_experiments_results.ipynb`: Notebook with the paper plots from the experiment data.


## Setup
`uv` is recommended but standard `pip` works too.
```bash
uv pip install -e ".[dev]"
# or
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Recipes

- **Joint Training and Modular Training (Hydra)**
  ```bash
  uv run src/jssp_agv/train_gnn_aec.py 
  uv run src/jssp_agv/train_gnn_jssp.py 
  uv run src/jssp_agv/train_gnn_agv.py 
  ```

- **Run Experiment: Benchmark Comparison**
Requires the models to be defined in the solver_collection.py file.
  ```python
  uv run src/jssp_agv/benchmark/bm_launcher.py 
  ```
- **Run Experiment: Grid Experiment**
Requires the models to be defined in the solver_collection.py file
  ```python
  uv run src/jssp_agv/benchmark/grid_launcher.py 
  ```

## Key Components
- **Observation Providers**: includes observation providers for AGV and Job scheduler models.
- **Reward Functions**: registry includes `negative makespan`
- **Heuristics**: job rules (`spt`, `smpt`, `lpt`, `mwr`, `lwr`, `fddmwr`, `mor`, `lor`, `fcfs`, `random`) and AGV rules (`sput`, `scta`, `scpt`, `random`).


## Additional Resources
- `src/jssp_agv/SOLVERMAPPING.md`: contains a mapping of the solver name in the experiment data to the paper.



