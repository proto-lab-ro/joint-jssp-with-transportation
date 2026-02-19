"""
Unified Benchmark Runner for Comparing Multiple Solvers.

This script serves as a central entry point for running comprehensive benchmarks
comparing various ML models (GNN, LoRA, AGV) against a full suite of
heuristic solvers.


To run this script:
1. Populate the `GNN_MODEL_PATHS`, `LORA_MODEL_PATHS`, and `AGV_MODEL_PATHS`
   lists with the Hydra output directories of the models you want to compare.
2. Adjust the `BATCH_EXPERIMENTS_CONFIG` to control the benchmark's scale.
3. Run from the root directory:
   python -m src.jssp_agv.benchmark.comparison_runner
"""

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
    HEURISTIC_SOLVERS,
    JSSP_MODEL_PATHS,
    LORA_MODEL_PATHS,
)
from jssp_agv.benchmark.utils import (
    get_combined_jssp_and_agv_solvers,
    get_gnn_solvers,
    get_heuristic_solvers,
    summarize_solver_loading_metrics,
)
from jssp_agv.db import bm_db_client
from jssp_core.instances import RandomTransportInstanceGenerator


# --- Configuration ---

parser = argparse.ArgumentParser()
parser.add_argument("--num_jobs", type=int, required=True)
parser.add_argument("--num_machines", type=int, required=True)
parser.add_argument("--num_agvs", type=int, required=True)


args = parser.parse_args()
# Configuration for the batch experiments
BATCH_EXPERIMENTS_CONFIG = {
    "num_instances": 100,
    "num_jobs": args.num_jobs,
    "num_machines": args.num_machines,
    "num_agvs": args.num_agvs,
    "seed_start": 2025,
    "verbose": False,
}

INSTANCE_GENERATOR_CONFIG = {
    "min_duration": 1,
    "max_duration": 100,
    "num_jobs": BATCH_EXPERIMENTS_CONFIG["num_jobs"],
    "num_machines": BATCH_EXPERIMENTS_CONFIG["num_machines"],
}

instance_generator = RandomTransportInstanceGenerator(**INSTANCE_GENERATOR_CONFIG)
rm = instance_generator.generate()
# Path to save the final benchmark results and plots
# Automatically creates date-based folder structure: results/benchmarks/YYYY-MM-DD/HH-MM-SS/
timestamp = datetime.now()
date_folder = timestamp.strftime("%Y-%m-%d")
time_folder = timestamp.strftime("%H-%M-%S")
benchmark_dir = Path("results/benchmarks") / date_folder / time_folder

RESULTS_SAVE_PATH = benchmark_dir / "multi_solver_comparison.csv"
PLOT_SAVE_PATH = benchmark_dir / "multi_solver_comparison_plot.png"


def main():
    """
    Main function to initialize solvers and run the multi-solver benchmark.
    """
    print("=== Starting Unified Multi-Solver Benchmark ===")
    experiment_params = {
        "num_instances": BATCH_EXPERIMENTS_CONFIG["num_instances"],
        "num_jobs": BATCH_EXPERIMENTS_CONFIG["num_jobs"],
        "num_machines": BATCH_EXPERIMENTS_CONFIG["num_machines"],
        "num_agvs": BATCH_EXPERIMENTS_CONFIG["num_agvs"],
        "min_operation_duration": INSTANCE_GENERATOR_CONFIG["min_duration"],
        "max_operation_duration": INSTANCE_GENERATOR_CONFIG["max_duration"],
        "min_travel_time": INSTANCE_GENERATOR_CONFIG["min_duration"],
        "max_travel_time": INSTANCE_GENERATOR_CONFIG["max_duration"],
        "seed_start": BATCH_EXPERIMENTS_CONFIG["seed_start"],
    }
    # --- 2. Initialize All ML Solvers ---
    ml_solvers = []
    all_model_paths = [
        (GNN_MODEL_PATHS,),
        (LORA_MODEL_PATHS,),
        (AGV_MODEL_PATHS,),
        (JSSP_MODEL_PATHS,),
    ]

    ml_solvers, gnn_metrics = get_gnn_solvers(
        ml_solvers,
        GNN_MODEL_PATHS,
        BATCH_EXPERIMENTS_CONFIG,
        rm,
        bm_db_client,
        experiment_params,
    )

    # Adapted as combinations of JSSP and AGV models
    ml_solvers, combined_metrics = get_combined_jssp_and_agv_solvers(
        ml_solvers=ml_solvers,
        JSSP_MODEL_PATHS=JSSP_MODEL_PATHS,
        AGV_MODEL_PATHS=AGV_MODEL_PATHS,
        BATCH_EXPERIMENTS_CONFIG=BATCH_EXPERIMENTS_CONFIG,
        rm=rm,
        db_client=bm_db_client,
        experiment_params=experiment_params,
    )

    # --- 3. Initialize and Run Multi-Solver Batch Experiment ---
    # Prepare original model paths dictionary for metadata

    heuristic_solvers, heuristic_metrics = get_heuristic_solvers(
        HEURISTIC_SOLVERS,
        BATCH_EXPERIMENTS_CONFIG,
        rm,
        bm_db_client,
        experiment_params,
    )

    summarize_solver_loading_metrics(
        [gnn_metrics, combined_metrics, heuristic_metrics],
    )
    multi_batch_runner = UniversalBenchmarkModule(
        universal_solver=heuristic_solvers + ml_solvers,
        instance_generator=instance_generator,
    )

    print(f"\nRunning batch experiments: {BATCH_EXPERIMENTS_CONFIG}")

    results_df = multi_batch_runner.run_batch_experiments(
        **BATCH_EXPERIMENTS_CONFIG, save_path=RESULTS_SAVE_PATH
    )

    # --- 4. Generate and Save Plots ---
    if not results_df.empty:
        try:
            multi_batch_runner.plot_results(
                results_df=results_df,
                save_path=PLOT_SAVE_PATH,
                show_plot=False,  # Set to True if you want to see the plot interactively
            )
            print(f"Plots saved to {PLOT_SAVE_PATH}")
        except Exception as e:
            print(f"Warning: Could not generate plots. Error: {e}")

    print("\n=== Benchmark Complete ===")

    print("\n=== Saving Results into Database ===")

    experiment_id = bm_db_client.insert_experiment_if_unique(experiment_params)
    print("Experiment ID:", experiment_id)

    with open(RESULTS_SAVE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        bm_db_client.insert_batch_results(reader, experiment_id)


if __name__ == "__main__":
    main()
