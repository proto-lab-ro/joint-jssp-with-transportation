import argparse
import csv
from datetime import datetime
from pathlib import Path

from jssp_agv.benchmark.modules import (
    UniversalBenchmarkModule,
)
from jssp_agv.benchmark.solver_collection import (
    AGV_MODEL_PATHS,
    GNN_MODEL_PATHS,
    JSSP_MODEL_PATHS,
    LORA_MODEL_PATHS,
)
from jssp_agv.benchmark.utils import (
    get_combined_jssp_and_agv_solvers,
    get_gnn_solvers,
    # get_heuristic_solvers,
    get_mor_and_scta_heuristic_solvers,
    summarize_solver_loading_metrics,
)
from jssp_agv.db import grid_db_client
from jssp_core import set_seed
from jssp_core.instances.generators import (
    GridRandomTransportInstanceGenerator,
)


set_seed(42)
# --- Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument("--num_jobs", type=int, required=True)
parser.add_argument("--num_machines", type=int, required=True)
parser.add_argument("--num_agvs", type=int, required=True)
parser.add_argument("--interval", type=int, required=True)
parser.add_argument("--std", type=float, required=True)
parser.add_argument("--num_instances", type=int, required=True)
# Lists of model paths to compare.
# Each path should point to the hydra output directory of a trained model.

args = parser.parse_args()
# Configuration for the batch experiments
GRID_EXPERIMENTS_CONFIG = {
    "num_instances": args.num_instances,
    "num_jobs": args.num_jobs,
    "num_machines": args.num_machines,
    "num_agvs": args.num_agvs,
    "max_duration": 100,
    "number_cells": 10,
    "seed_start": 2025,
    "verbose": False,
    "box_max_process_time": False,
    "interval": args.interval,
    "std": args.std,
    "grid_type": "uniform10",  # truncated or norm_truncated or uniform
}

INSTANCE_GENERATOR_CONFIG = {
    "num_jobs": GRID_EXPERIMENTS_CONFIG["num_jobs"],
    "num_machines": GRID_EXPERIMENTS_CONFIG["num_machines"],
}


instance_generator = GridRandomTransportInstanceGenerator(**INSTANCE_GENERATOR_CONFIG)

rm = instance_generator.generate(
    min_op_duration=1,
    min_travel_duration=1,
    max_op_duration=100,
    max_travel_duration=100,
)
# Path to save the final benchmark results and plots
# Automatically creates date-based folder structure: results/benchmarks/YYYY-MM-DD/HH-MM-SS/
timestamp = datetime.now()
date_folder = timestamp.strftime("%Y-%m-%d")
time_folder = timestamp.strftime("%H-%M-%S")
benchmark_dir = Path("results/benchmarks") / date_folder / time_folder

RESULTS_SAVE_PATH = benchmark_dir / "multi_solver_comparison.csv"
PLOT_SAVE_PATH = benchmark_dir / "multi_solver_comparison_plot.png"


# --- Main Execution ---


def main():
    """
    Main function to initialize solvers and run the multi-solver benchmark.
    """
    print("=== Starting Unified Multi-Solver Benchmark ===")
    experiment_params = {
        "num_instances": GRID_EXPERIMENTS_CONFIG["num_instances"],
        "num_jobs": GRID_EXPERIMENTS_CONFIG["num_jobs"],
        "num_machines": GRID_EXPERIMENTS_CONFIG["num_machines"],
        "num_agvs": GRID_EXPERIMENTS_CONFIG["num_agvs"],
        "max_duration": GRID_EXPERIMENTS_CONFIG["max_duration"],
        "number_cells": GRID_EXPERIMENTS_CONFIG["number_cells"],
        "interval": GRID_EXPERIMENTS_CONFIG["interval"],
        "std": GRID_EXPERIMENTS_CONFIG["std"],
        "grid_type": GRID_EXPERIMENTS_CONFIG["grid_type"],
        "seed_start": GRID_EXPERIMENTS_CONFIG["seed_start"],
    }
    # --- 2. Initialize All ML Solvers ---
    ml_solvers = []
    all_model_paths = [
        (GNN_MODEL_PATHS,),
        (LORA_MODEL_PATHS,),
        (AGV_MODEL_PATHS,),
    ]
    ml_solvers, gnn_metrics = get_gnn_solvers(
        ml_solvers,
        GNN_MODEL_PATHS,
        GRID_EXPERIMENTS_CONFIG,
        rm,
        grid_db_client,
        experiment_params,
    )

    ml_solvers, combined_metrics = get_combined_jssp_and_agv_solvers(
        ml_solvers=ml_solvers,
        JSSP_MODEL_PATHS=JSSP_MODEL_PATHS,
        AGV_MODEL_PATHS=AGV_MODEL_PATHS,
        BATCH_EXPERIMENTS_CONFIG=GRID_EXPERIMENTS_CONFIG,
        rm=rm,
        db_client=grid_db_client,
        experiment_params=experiment_params,
    )

    heuristic_solvers, heuristic_metrics = get_mor_and_scta_heuristic_solvers(
        GRID_EXPERIMENTS_CONFIG,
        rm,
        grid_db_client,
        experiment_params,
    )

    summarize_solver_loading_metrics(
        [gnn_metrics, combined_metrics, heuristic_metrics],  # heuristic_metrics
    )

    multi_batch_runner = UniversalBenchmarkModule(
        universal_solver=ml_solvers + heuristic_solvers,
        instance_generator=instance_generator,
    )

    print(f"\nRunning batch experiments: {GRID_EXPERIMENTS_CONFIG}")

    results_df = multi_batch_runner.run_grid_experiments(
        **GRID_EXPERIMENTS_CONFIG, save_path=RESULTS_SAVE_PATH
    )

    print("\n=== Benchmark Complete ===")

    print("\n=== Saving Results into Database ===")

    experiment_id = grid_db_client.insert_experiment_if_unique(experiment_params)
    print("Experiment ID:", experiment_id)

    with open(RESULTS_SAVE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        grid_db_client.insert_batch_results(reader, experiment_id)


if __name__ == "__main__":
    main()
