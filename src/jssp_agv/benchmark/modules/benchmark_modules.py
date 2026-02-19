from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from jssp_agv.benchmark.modules.universal_solver import UniversalSolver
from jssp_core.instances import InstanceGenerator


class UniversalBenchmarkModule:
    """
    Batch experimentation framework for comparing multiple ML solvers against heuristics.


    Example:
        universal_solver = UniversalSolver(

        # Initialize multiple ML solvers
        ml_solver_1 = TransportMLSolver(...)
        ml_solver_2 = TransportMLSolver(...)

        batch_exp = MultiSolverBatchExperiments(
            heuristic_solver=heuristic_solver,
            ml_solvers=[ml_solver_1, ml_solver_2]
        )

        results = batch_exp.run_batch_experiments(
            num_instances=10,
            num_jobs=6,
            num_machines=6,
            num_agvs=3
        )
    """

    def __init__(
        self,
        universal_solver: list[UniversalSolver] = None,
        instance_generator: InstanceGenerator | None = None,
    ):
        """
        Initialize MultiSolverBatchExperiments.

        Parameters
        ----------
        heuristic_solver : TransportSolver
            The heuristic-based solver.
        ml_solvers : List[TransportMLSolver]
            A list of ML-based solvers to compare.
        original_model_paths : Optional[Dict[str, List[str]]]
            Dictionary mapping model types (e.g., 'GNN', 'LORA', 'AGV') to their original paths.
        """

        self.universal_solver = universal_solver
        self.results_history = []
        self.instance_generator = instance_generator

        # Store model paths for reference
        # self.model_paths = {
        #     solver.get_name(): str(solver.model_path) for solver in universal_solver
        # }

    def run_batch_experiments(
        self,
        num_instances: int,
        num_jobs: int,
        num_machines: int,
        num_agvs: int = 3,
        min_operation_duration: int = 1,
        max_operation_duration: int = 100,
        min_travel_time: int = 1,
        max_travel_time: int = 100,
        seed_start: int | None = None,
        verbose: bool = True,
        save_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Run batch experiments comparing all heuristics and a list of ML solvers.

        Generates multiple random instances and solves each with:
        1. All heuristic combinations (using evaluate_all_heuristics)
        2. ML solver (trained policy)

        Parameters
        ----------
        num_instances : int
            Number of random instances to generate
        num_jobs : int
            Number of jobs per instance
        num_machines : int
            Number of machines per instance
        num_agvs : int
            Number of AGVs to use (default: 3)
        min_operation_duration : int
            Minimum operation duration (default: 1)
        max_operation_duration : int
            Maximum operation duration (default: 10)
        min_travel_time : int
            Minimum travel time between machines (default: 1)
        max_travel_time : int
            Maximum travel time between machines (default: 5)
        seed_start : Optional[int]
            Starting seed for random generation. If None, uses random seeds
        verbose : bool
            Whether to print progress (default: True)
        save_path : Optional[Path]
            Path to save results CSV. If None, auto-generates path based on model directory

        Returns
        -------
        pd.DataFrame
            Results with columns: instance_id, solver_type, solver_name, job_heuristic,
            agv_heuristic, makespan, seed
        """
        results = []

        if verbose:
            print(f"Running batch experiments with {len(self.ml_solvers)} ML solvers")

        for i in range(num_instances):
            seed = seed_start + i if seed_start is not None else None
            instance = self.instance_generator.generate(seed=seed)

            progress_pct = (i + 1) / num_instances * 100
            prev_pct = i / num_instances * 100
            if int(progress_pct / 10) > int(prev_pct / 10) or i == 0:
                print(
                    f"Progress: {int(progress_pct)}% ({i + 1}/{num_instances} instances)"
                )

            # 2. Solve with each ML solver
            for solver in self.universal_solver:
                # The num_agvs is part of the instance data, so not needed here
                makespan, schedule_ml = solver.solve(
                    instance=instance, num_agvs=num_agvs
                )
                j_p, a_p = solver.get_model_path()
                agv_solver_type = "ml" if solver.agv_is_ml else "heuristic"
                job_solver_type = "ml" if solver.job_is_ml else "heuristic"

                solver_type = (
                    "ml"
                    if solver.agv_is_ml and solver.job_is_ml
                    else "heuristic"
                    if not solver.agv_is_ml and not solver.job_is_ml
                    else "mixture"
                )
                results.append(
                    {
                        "instance_id": i,
                        "agv_solver_type": agv_solver_type,
                        "job_solver_type": job_solver_type,
                        "solver_type": solver_type,
                        "solver_name": solver.get_name(),
                        "makespan": makespan,
                        "job_model": j_p,
                        "agv_model": a_p,
                        "seed": seed,
                    }
                )

        results_df = pd.DataFrame(results)
        self.results_history.append(results_df)

        if verbose:
            self._print_summary_statistics(results_df)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(save_path, index=False)

            # Save experiment configuration and model paths metadata
            metadata_path = save_path.parent / "experiment_config.txt"
            with open(metadata_path, "w") as f:
                f.write("Batch Experiment Configuration\n")
                f.write("=" * 80 + "\n\n")

                # Write experiment parameters
                f.write("Experiment Parameters:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  num_instances: {num_instances}\n")
                f.write(f"  num_jobs: {num_jobs}\n")
                f.write(f"  num_machines: {num_machines}\n")
                f.write(f"  num_agvs: {num_agvs}\n")
                f.write(f"  min_operation_duration: {min_operation_duration}\n")
                f.write(f"  max_operation_duration: {max_operation_duration}\n")
                f.write(f"  min_travel_time: {min_travel_time}\n")
                f.write(f"  max_travel_time: {max_travel_time}\n")
                f.write(f"  seed_start: {seed_start}\n")
                f.write("\n" + "=" * 80 + "\n\n")

                f.write("ML Solver Model Paths\n")
                f.write("=" * 80 + "\n\n")

                # Write original configuration paths if available
                # if self.original_model_paths:
                #     f.write("Original Configuration Paths:\n")
                #     f.write("-" * 80 + "\n")
                #     for model_type, paths in self.original_model_paths.items():
                #         if paths:  # Only write if there are paths
                #             f.write(f"\n{model_type}:\n")
                #             for path in paths:
                #                 f.write(f"  - {path}\n")
                #     f.write("\n" + "=" * 80 + "\n\n")

                # Write loaded solver model paths
                # f.write("Loaded Solver Model Paths:\n")
                # f.write("-" * 80 + "\n")
                # for solver_name, model_path in self.model_paths.items():
                #     f.write(f"\n{solver_name}:\n  {model_path}\n")

            if verbose:
                print(f"Results saved to: {save_path}")
                print(f"Experiment config saved to: {metadata_path}")

        return results_df

    def _set_up_grid(
        self,
        max_duration: int,
        number_cells: int,
        num_instances: int,
        verbose: bool = False,
    ):
        """Set up UUP grid parameters for experiments."""
        # Create linspaces for min values (starting from 1 to avoid zero durations)
        min_op_linspace = np.linspace(1, 91, number_cells).astype(int)
        min_travel_linspace = np.linspace(1, 91, number_cells).astype(int)

        # Ensure unique values and proper ordering
        min_op_linspace = np.unique(min_op_linspace)
        min_travel_linspace = np.unique(min_travel_linspace)

        # Create meshgrid
        op_grid, travel_grid = np.meshgrid(min_op_linspace, min_travel_linspace)

        if verbose:
            print("Running grid experiments")
            print(f"  Min operation values: {min_op_linspace}")
            print(f"  Min travel values: {min_travel_linspace}")
            print(f"  Max duration (both): {max_duration}")
            print(f"  Grid size: {op_grid.shape[0]} x {op_grid.shape[1]}")
            print(f"  Instances per grid point: {num_instances}")
            print(f"  Total instances: {op_grid.size * num_instances}")
            print(f"  ML solvers: {len(self.universal_solver)}")
            print("=" * 60)

        return op_grid, travel_grid

    def __set_up_grid(
        self,
        max_duration: int,
        number_cells: int,
        num_instances: int,
        verbose: bool = False,
    ):
        """Set up UUP grid parameters for experiments."""
        # Create linspaces for min values (starting from 1 to avoid zero durations)
        min_op_linspace = np.linspace(1, max_duration - 1, number_cells).astype(int)
        min_travel_linspace = np.linspace(1, max_duration - 1, number_cells).astype(int)

        # Ensure unique values and proper ordering
        min_op_linspace = np.unique(min_op_linspace)
        min_travel_linspace = np.unique(min_travel_linspace)

        # Create meshgrid
        op_grid, travel_grid = np.meshgrid(min_op_linspace, min_travel_linspace)

        if verbose:
            print("Running grid experiments")
            print(f"  Min operation values: {min_op_linspace}")
            print(f"  Min travel values: {min_travel_linspace}")
            print(f"  Max duration (both): {max_duration}")
            print(f"  Grid size: {op_grid.shape[0]} x {op_grid.shape[1]}")
            print(f"  Instances per grid point: {num_instances}")
            print(f"  Total instances: {op_grid.size * num_instances}")
            print(f"  ML solvers: {len(self.universal_solver)}")
            print("=" * 60)

        return op_grid, travel_grid

    def run_grid_experiments(
        self,
        num_instances: int,
        num_jobs: int,
        num_machines: int,
        num_agvs: int = 3,
        max_duration: int = 100,
        number_cells: int = 1,
        seed_start: int | None = None,
        verbose: bool = True,
        save_path: Path | None = None,
        box_max_process_time=False,
        grid_type: str = "uniform10",
        interval: int = 10.0,
        std: int = 5.0,
    ) -> pd.DataFrame:
        """
        Run grid batch experiments comparing all heuristics and a list of ML solvers.

        Generates multiple random instances and solves each with:
        1. All heuristic combinations (using evaluate_all_heuristics)
        2. ML solver (trained policy)

        Parameters
        ----------
        num_instances : int
            Number of random instances to generate
        num_jobs : int
            Number of jobs per instance
        num_machines : int
            Number of machines per instance
        num_agvs : int
            Number of AGVs to use (default: 3)
        min_operation_duration : int
            Minimum operation duration (default: 1)
        max_operation_duration : int
            Maximum operation duration (default: 10)
        min_travel_time : int
            Minimum travel time between machines (default: 1)
        max_travel_time : int
            Maximum travel time between machines (default: 5)
        seed_start : Optional[int]
            Starting seed for random generation. If None, uses random seeds
        verbose : bool
            Whether to print progress (default: True)
        save_path : Optional[Path]
            Path to save results CSV. If None, auto-generates path based on model directory

        Returns
        -------
        pd.DataFrame
            Results with columns: instance_id, solver_type, solver_name, job_heuristic,
            agv_heuristic, makespan, seed
        """

        op_grid, travel_grid = self._set_up_grid(
            max_duration=max_duration,
            number_cells=number_cells,
            num_instances=num_instances,
            verbose=verbose,
        )
        heatmap_data = []
        total_points = op_grid.size
        point_idx = 0
        results = []
        total_number_cells = op_grid.shape[0] * op_grid.shape[1]
        tqdm_bar = tqdm(total=total_number_cells, desc="Grid Points", unit="point")

        if grid_type == "uniform10":
            results, heatmap_data = self._uniform_grid(
                op_grid=op_grid,
                travel_grid=travel_grid,
                max_duration=max_duration,
                num_instances=num_instances,
                seed_start=seed_start,
                box_max_process_time=box_max_process_time,
                num_agvs=num_agvs,
                results=results,
                heatmap_data=heatmap_data,
                tqdm_bar=tqdm_bar,
                point_idx=point_idx,
            )

        elif grid_type == "truncated" or grid_type == "norm_truncated":
            results, heatmap_data = self._truncated_grid(
                op_grid=op_grid,
                travel_grid=travel_grid,
                max_duration=max_duration,
                num_instances=num_instances,
                seed_start=seed_start,
                box_max_process_time=box_max_process_time,
                num_agvs=num_agvs,
                results=results,
                heatmap_data=heatmap_data,
                tqdm_bar=tqdm_bar,
                point_idx=point_idx,
            )

        else:
            raise ValueError(f"Unknown grid_type: {grid_type}")

        tqdm_bar.close()

        # Create DataFrames
        results_df = pd.DataFrame(results)
        heatmap_df = pd.DataFrame(heatmap_data)

        self.results_history.append(results_df)

        if verbose:
            print("\n" + "=" * 60)
            print("Grid experiments complete!")
            print(f"Raw results: {len(results_df)} rows")
            print(f"Heatmap data: {len(heatmap_df)} rows")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(save_path, index=False)

            # Save heatmap data separately
            heatmap_path = save_path.parent / "grid_heatmap_data.csv"
            heatmap_df.to_csv(heatmap_path, index=False)

            # Save experiment configuration and model paths metadata
            metadata_path = save_path.parent / "grid_experiment_metadata.txt"
            with open(metadata_path, "w") as f:
                f.write("Batch Experiment Configuration\n")
                f.write("=" * 80 + "\n\n")

                # Write experiment parameters
                f.write("Experiment Parameters:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  num_instances: {num_instances}\n")
                f.write(f"  num_jobs: {num_jobs}\n")
                f.write(f"  num_machines: {num_machines}\n")
                f.write(f"  num_agvs: {num_agvs}\n")
                f.write(f"  seed_start: {seed_start}\n")
                f.write(f"  max_duration: {max_duration}\n")
                f.write(f"  number_cells: {number_cells}\n")
                f.write(f"  grid_type: {grid_type}\n")
                f.write(f"  interval: {interval}\n")
                f.write(f"  std: {std}\n")
                f.write("\n" + "=" * 80 + "\n\n")

                f.write("ML Solver Model Paths\n")
                f.write("=" * 80 + "\n\n")

            if verbose:
                print(f"Results saved to: {save_path}")
                print(f"Experiment config saved to: {metadata_path}")

        return results_df

    def _truncated_grid(
        self,
        op_grid,
        travel_grid,
        max_duration,
        num_instances,
        seed_start,
        box_max_process_time,
        num_agvs,
        results,
        heatmap_data,
        tqdm_bar,
        point_idx,
    ):
        """Run the uniform grid analysis"""
        for i in range(op_grid.shape[0]):
            for j in range(op_grid.shape[1]):
                min_op = int(op_grid[i, j])
                min_travel = int(travel_grid[i, j])
                point_idx += 1

                for nr_inst in range(num_instances):
                    seed = seed_start + nr_inst if seed_start is not None else None
                    instance = self.instance_generator.generate(
                        seed=seed,
                        min_op_duration=min_op,
                        min_transport_duration=min_travel,
                    )

                    # 2. Solve with each ML solver
                    for solver in self.universal_solver:
                        result_dict = self._apply_solver_to_instance(
                            solver,
                            instance,
                            num_agvs,
                            nr_inst,
                            min_travel,
                            min_op,
                            seed,
                        )
                        results.append(result_dict)
                tqdm_bar.update(1)

        return results, heatmap_data

    def _uniform_grid(
        self,
        op_grid,
        travel_grid,
        max_duration,
        num_instances,
        seed_start,
        box_max_process_time,
        num_agvs,
        results,
        heatmap_data,
        tqdm_bar,
        point_idx,
    ):
        """Run the uniform grid analysis"""
        for i in range(op_grid.shape[0]):
            for j in range(op_grid.shape[1]):
                min_op = int(op_grid[i, j])
                min_travel = int(travel_grid[i, j])
                point_idx += 1
                op_next_cell = None
                travel_next_cell = None
                if j < op_grid.shape[1] - 1:
                    op_next_cell = int(op_grid[i, j + 1])
                else:
                    op_next_cell = 101

                if i < travel_grid.shape[0] - 1:
                    travel_next_cell = int(travel_grid[i + 1, j])
                else:
                    travel_next_cell = 101
                op_next_cell -= 1
                travel_next_cell -= 1
                for nr_inst in range(num_instances):
                    # seed = seed_start + nr_inst if seed_start is not None else None
                    instance = self.instance_generator.generate(
                        # seed=seed,
                        min_op_duration=min_op,
                        max_op_duration=op_next_cell,
                        min_travel_duration=min_travel,
                        max_travel_duration=travel_next_cell,
                    )
                    # print(
                    #     f"Generated instance with min_op: {min_op}, max_op: {op_next_cell}, "
                    #     f"min_travel: {min_travel}, max_travel: {travel_next_cell}"
                    # )
                    # print(instance)

                    # 2. Solve with each ML solver
                    for solver in self.universal_solver:
                        result_dict = self._apply_solver_to_instance(
                            solver,
                            instance,
                            num_agvs,
                            nr_inst,
                            min_travel,
                            min_op,
                            42,
                        )
                        results.append(result_dict)
                tqdm_bar.update(1)

        return results, heatmap_data

    def __uniform_grid(
        self,
        op_grid,
        travel_grid,
        max_duration,
        num_instances,
        seed_start,
        box_max_process_time,
        num_agvs,
        results,
        heatmap_data,
        tqdm_bar,
        point_idx,
    ):
        """Run the uniform grid analysis"""
        for i in range(op_grid.shape[0]):
            for j in range(op_grid.shape[1]):
                min_op = int(op_grid[i, j])
                min_travel = int(travel_grid[i, j])
                point_idx += 1
                op_next_cell = None
                travel_next_cell = None
                if j < op_grid.shape[1] - 1:
                    op_next_cell = int(op_grid[i, j + 1])
                else:
                    op_next_cell = max_duration

                if i < travel_grid.shape[0] - 1:
                    travel_next_cell = int(travel_grid[i + 1, j])
                else:
                    travel_next_cell = max_duration

                for nr_inst in range(num_instances):
                    # seed = seed_start + nr_inst if seed_start is not None else None
                    instance = self.instance_generator.generate(
                        # seed=seed,
                        min_op_duration=min_op,
                        max_op_duration=op_next_cell,
                        min_travel_duration=min_travel,
                        max_travel_duration=travel_next_cell,
                    )
                    # print(
                    #     f"Generated instance with min_op: {min_op}, max_op: {op_next_cell}, "
                    #     f"min_travel: {min_travel}, max_travel: {travel_next_cell}"
                    # )
                    # print(instance)

                    # 2. Solve with each ML solver
                    for solver in self.universal_solver:
                        result_dict = self._apply_solver_to_instance(
                            solver,
                            instance,
                            num_agvs,
                            nr_inst,
                            min_travel,
                            min_op,
                            42,
                        )
                        results.append(result_dict)
                tqdm_bar.update(1)

        return results, heatmap_data

    def _apply_solver_to_instance(
        self, solver, instance, num_agvs, nr_inst, min_travel, min_op, seed
    ):
        makespan, schedule_ml = solver.solve(instance=instance, num_agvs=num_agvs)
        j_p, a_p = solver.get_model_path()
        agv_solver_type = "ml" if solver.agv_is_ml else "heuristic"
        job_solver_type = "ml" if solver.job_is_ml else "heuristic"

        solver_type = (
            "ml"
            if solver.agv_is_ml and solver.job_is_ml
            else "heuristic"
            if not solver.agv_is_ml and not solver.job_is_ml
            else "mixture"
        )

        result_dict = {
            "instance_id": nr_inst,
            "agv_solver_type": agv_solver_type,
            "job_solver_type": job_solver_type,
            "solver_type": solver_type,
            "solver_name": solver.get_name(),
            "makespan": makespan,
            "job_model": j_p,
            "agv_model": a_p,
            "seed": seed,
            "nr_inst": nr_inst,
            "min_travel": min_travel,
            "min_op": min_op,
            "op2transport_ratio": self._compute_op2transport_ratio(min_travel, min_op),
        }

        return result_dict

    def _compute_op2transport_ratio(self, min_travel, min_op):
        min_val = 0
        max_val = 100

        # Step 1: Scale individually to [0,1]
        op_scaled = (min_op - min_val) / (max_val - min_val)
        tr_scaled = (min_travel - min_val) / (max_val - min_val)

        # Step 2: Compute normalized ratio
        ratios_norm = op_scaled / (op_scaled + tr_scaled)

        # Step 3: Rescale to [-1, 1]
        ratios_scaled = 2 * ratios_norm - 1
        return ratios_scaled

    def plot_grid_results(
        self,
        heatmap_df: pd.DataFrame,
        metric: str = "mean",
        save_path: Path | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plot heatmap results from grid experiments.

        Creates heatmaps showing solver performance across the grid of
        operation and travel time parameters.

        Parameters
        ----------
        heatmap_df : pd.DataFrame
            DataFrame from run_grid_experiments with heatmap data
        metric : str
            Which metric to plot: 'min', 'max', or 'mean' (default: 'mean')
        save_path : Optional[Path]
            Path to save the plot
        show_plot : bool
            Whether to display the plot (default: True)
        """
        if heatmap_df.empty:
            print("Cannot plot grid results. DataFrame is empty.")
            return

        # Filter by metric
        metric_df = heatmap_df[heatmap_df["metric"] == metric]

        if metric_df.empty:
            print(f"No data for metric '{metric}'")
            return

        # Get unique solvers
        solver_names = metric_df["solver_name"].unique()
        n_solvers = len(solver_names)

        # Create subplots
        n_cols = min(3, n_solvers)
        n_rows = (n_solvers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        if n_solvers == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # Find global min/max for consistent colorbar
        vmin = metric_df["makespan"].min()
        vmax = metric_df["makespan"].max()

        for idx, solver_name in enumerate(solver_names):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            solver_data = metric_df[metric_df["solver_name"] == solver_name]

            # Pivot for heatmap
            pivot = solver_data.pivot(
                index="min_travel", columns="min_operation", values="makespan"
            )

            # Sort index and columns
            pivot = pivot.sort_index(ascending=False)  # Higher travel at top
            pivot = pivot.sort_index(axis=1)  # Lower operation at left

            sns.heatmap(
                pivot,
                ax=ax,
                cmap="RdYlGn_r",  # Red=high (bad), Green=low (good)
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=".0f",
                cbar_kws={"label": "Makespan"},
            )

            ax.set_title(f"{solver_name}\n({metric} makespan)", fontsize=12)
            ax.set_xlabel("Min Operation Duration", fontsize=10)
            ax.set_ylabel("Min Travel Time", fontsize=10)

        # Hide unused subplots
        for idx in range(n_solvers, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(
            f"Grid Experiment Results ({metric.capitalize()} Makespan)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

        plt.close(fig)

    def plot_grid_comparison(
        self,
        heatmap_df: pd.DataFrame,
        metric: str = "mean",
        save_path: Path | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plot a comparison heatmap showing ML solver gap vs best heuristic.

        Parameters
        ----------
        heatmap_df : pd.DataFrame
            DataFrame from run_grid_experiments with heatmap data
        metric : str
            Which metric to compare: 'min', 'max', or 'mean' (default: 'mean')
        save_path : Optional[Path]
            Path to save the plot
        show_plot : bool
            Whether to display the plot (default: True)
        """
        if heatmap_df.empty:
            print("Cannot plot comparison. DataFrame is empty.")
            return

        metric_df = heatmap_df[heatmap_df["metric"] == metric]

        # Get best heuristic data
        heur_df = metric_df[metric_df["solver_name"] == "best_heuristic"]
        if heur_df.empty:
            print("No best_heuristic data found.")
            return

        # Get ML solver names
        ml_solvers = metric_df[metric_df["solver_type"] == "ml"]["solver_name"].unique()
        if len(ml_solvers) == 0:
            print("No ML solver data found.")
            return

        n_cols = min(3, len(ml_solvers))
        n_rows = (len(ml_solvers) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        if len(ml_solvers) == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, solver_name in enumerate(ml_solvers):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            ml_data = metric_df[metric_df["solver_name"] == solver_name]

            # Merge to compute gap
            merged = ml_data.merge(
                heur_df[["min_operation", "min_travel", "makespan"]],
                on=["min_operation", "min_travel"],
                suffixes=("_ml", "_heur"),
            )

            merged["gap_pct"] = (
                (merged["makespan_ml"] - merged["makespan_heur"])
                / merged["makespan_heur"]
                * 100
            )

            # Pivot for heatmap
            pivot = merged.pivot(
                index="min_travel", columns="min_operation", values="gap_pct"
            )
            pivot = pivot.sort_index(ascending=False)
            pivot = pivot.sort_index(axis=1)

            # Symmetric colormap centered at 0
            abs_max = max(abs(pivot.min().min()), abs(pivot.max().max()))

            sns.heatmap(
                pivot,
                ax=ax,
                cmap="RdYlGn_r",  # Green=negative (ML better), Red=positive (heuristic better)
                center=0,
                vmin=-abs_max,
                vmax=abs_max,
                annot=True,
                fmt=".1f",
                cbar_kws={"label": "Gap (%)"},
            )

            ax.set_title(f"{solver_name}\nvs Best Heuristic ({metric})", fontsize=12)
            ax.set_xlabel("Min Operation Duration", fontsize=10)
            ax.set_ylabel("Min Travel Time", fontsize=10)

        # Hide unused subplots
        for idx in range(len(ml_solvers), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(
            f"ML Solver Gap vs Best Heuristic ({metric.capitalize()})\nNegative = ML Better, Positive = Heuristic Better",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

        plt.close(fig)

    def _print_ratio_summary_statistics(self, results_df: pd.DataFrame) -> None:
        """Print summary statistics for ratio experiment results."""
        print("\n" + "=" * 60)
        print("=== Ratio Experiment Summary ===")
        print("=" * 60)

        ratios = sorted(results_df["ratio"].unique())

        for ratio in ratios:
            ratio_df = results_df[results_df["ratio"] == ratio]
            heuristic_results = ratio_df[ratio_df["solver_type"] == "heuristic"]
            ml_results = ratio_df[ratio_df["solver_type"] == "ml"]

            print(f"\n--- Ratio {ratio} ---")

            if not heuristic_results.empty:
                best_per_instance = heuristic_results.loc[
                    heuristic_results.groupby("instance_id")["makespan"].idxmin()
                ]
                print(
                    f"  Best Heuristic Avg Makespan: {best_per_instance['makespan'].mean():.2f}"
                )

            if not ml_results.empty:
                for name, group in ml_results.groupby("solver_name"):
                    avg_makespan = group["makespan"].mean()
                    print(f"  {name} Avg Makespan: {avg_makespan:.2f}")

                    if not heuristic_results.empty:
                        improvement = (
                            best_per_instance["makespan"].mean() - avg_makespan
                        ) / best_per_instance["makespan"].mean()
                        print(f"    vs Best Heuristic: {improvement:+.2%}")

    def plot_ratio_results(
        self,
        results_df: pd.DataFrame,
        save_path: Path | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plot results across different JSSP-to-AGV ratios.

        Creates a multi-panel visualization showing:
        1. Boxplot of makespan by ratio for each solver type
        2. Line plot of average makespan vs ratio
        3. Win rate of ML solvers vs best heuristic across ratios

        Parameters
        ----------
        results_df : pd.DataFrame
            Results DataFrame with 'ratio' column from run_ratio_experiments
        save_path : Optional[Path]
            Path to save the plot
        show_plot : bool
            Whether to display the plot (default: True)
        """
        if results_df.empty or "ratio" not in results_df.columns:
            print("Cannot plot ratio results. Ensure DataFrame has 'ratio' column.")
            return

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ratios = sorted(results_df["ratio"].unique())
        heuristic_results = results_df[results_df["solver_type"] == "heuristic"]
        ml_results = results_df[results_df["solver_type"] == "ml"]

        # --- Plot 1: Average makespan vs ratio (line plot) ---
        ax1 = axes[0, 0]

        # Best heuristic per instance
        best_heur_avg = []
        for ratio in ratios:
            ratio_heur = heuristic_results[heuristic_results["ratio"] == ratio]
            best_per_instance = ratio_heur.loc[
                ratio_heur.groupby("instance_id")["makespan"].idxmin()
            ]
            best_heur_avg.append(best_per_instance["makespan"].mean())

        ax1.plot(
            ratios,
            best_heur_avg,
            "o-",
            label="Best Heuristic",
            linewidth=2,
            markersize=8,
        )

        # ML solvers
        ml_solver_names = sorted(ml_results["solver_name"].unique())
        colors = sns.color_palette("Set2", n_colors=len(ml_solver_names))

        for idx, solver_name in enumerate(ml_solver_names):
            solver_avg = []
            for ratio in ratios:
                ratio_ml = ml_results[
                    (ml_results["ratio"] == ratio)
                    & (ml_results["solver_name"] == solver_name)
                ]
                solver_avg.append(ratio_ml["makespan"].mean())
            ax1.plot(
                ratios,
                solver_avg,
                "s--",
                label=solver_name,
                linewidth=2,
                markersize=8,
                color=colors[idx],
            )

        ax1.set_xlabel("JSSP-to-AGV Ratio", fontsize=12)
        ax1.set_ylabel("Average Makespan", fontsize=12)
        ax1.set_title(
            "Average Makespan vs JSSP-to-AGV Ratio", fontsize=14, fontweight="bold"
        )
        ax1.legend()
        ax1.set_xscale("log")
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Boxplot by ratio ---
        ax2 = axes[0, 1]

        # Prepare data for boxplot
        plot_data = []
        for ratio in ratios:
            ratio_heur = heuristic_results[heuristic_results["ratio"] == ratio]
            best_per_instance = ratio_heur.loc[
                ratio_heur.groupby("instance_id")["makespan"].idxmin()
            ]
            for makespan in best_per_instance["makespan"]:
                plot_data.append(
                    {
                        "Ratio": str(ratio),
                        "Solver": "Best Heuristic",
                        "Makespan": makespan,
                    }
                )

            for solver_name in ml_solver_names:
                ratio_ml = ml_results[
                    (ml_results["ratio"] == ratio)
                    & (ml_results["solver_name"] == solver_name)
                ]
                for makespan in ratio_ml["makespan"]:
                    plot_data.append(
                        {
                            "Ratio": str(ratio),
                            "Solver": solver_name,
                            "Makespan": makespan,
                        }
                    )

        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x="Ratio", y="Makespan", hue="Solver", ax=ax2)
        ax2.set_xlabel("JSSP-to-AGV Ratio", fontsize=12)
        ax2.set_ylabel("Makespan", fontsize=12)
        ax2.set_title("Makespan Distribution by Ratio", fontsize=14, fontweight="bold")
        ax2.legend(title="Solver", bbox_to_anchor=(1.02, 1), loc="upper left")

        # --- Plot 3: Win rate vs ratio ---
        ax3 = axes[1, 0]

        for idx, solver_name in enumerate(ml_solver_names):
            win_rates = []
            for ratio in ratios:
                ratio_heur = heuristic_results[heuristic_results["ratio"] == ratio]
                best_per_instance = (
                    ratio_heur.groupby("instance_id")["makespan"].min().reset_index()
                )
                best_per_instance.columns = ["instance_id", "best_heuristic"]

                ratio_ml = ml_results[
                    (ml_results["ratio"] == ratio)
                    & (ml_results["solver_name"] == solver_name)
                ]

                merged = ratio_ml.merge(best_per_instance, on="instance_id")
                wins = (merged["makespan"] < merged["best_heuristic"]).sum()
                win_rate = 100 * wins / len(merged) if len(merged) > 0 else 0
                win_rates.append(win_rate)

            ax3.plot(
                ratios,
                win_rates,
                "o-",
                label=solver_name,
                linewidth=2,
                markersize=8,
                color=colors[idx],
            )

        ax3.axhline(
            y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold"
        )
        ax3.set_xlabel("JSSP-to-AGV Ratio", fontsize=12)
        ax3.set_ylabel("Win Rate vs Best Heuristic (%)", fontsize=12)
        ax3.set_title(
            "ML Solver Win Rate vs JSSP-to-AGV Ratio", fontsize=14, fontweight="bold"
        )
        ax3.legend()
        ax3.set_xscale("log")
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        # --- Plot 4: Gap percentage vs ratio ---
        ax4 = axes[1, 1]

        for idx, solver_name in enumerate(ml_solver_names):
            gap_pcts = []
            for ratio in ratios:
                ratio_heur = heuristic_results[heuristic_results["ratio"] == ratio]
                best_per_instance = (
                    ratio_heur.groupby("instance_id")["makespan"].min().reset_index()
                )
                best_per_instance.columns = ["instance_id", "best_heuristic"]

                ratio_ml = ml_results[
                    (ml_results["ratio"] == ratio)
                    & (ml_results["solver_name"] == solver_name)
                ]

                merged = ratio_ml.merge(best_per_instance, on="instance_id")
                gap_pct = (
                    (merged["makespan"] - merged["best_heuristic"])
                    / merged["best_heuristic"]
                    * 100
                ).mean()
                gap_pcts.append(gap_pct)

            ax4.plot(
                ratios,
                gap_pcts,
                "o-",
                label=solver_name,
                linewidth=2,
                markersize=8,
                color=colors[idx],
            )

        ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax4.set_xlabel("JSSP-to-AGV Ratio", fontsize=12)
        ax4.set_ylabel("Average Gap vs Best Heuristic (%)", fontsize=12)
        ax4.set_title(
            "ML Solver Gap vs JSSP-to-AGV Ratio", fontsize=14, fontweight="bold"
        )
        ax4.legend()
        ax4.set_xscale("log")
        ax4.grid(True, alpha=0.3)

        fig.suptitle(
            "Ratio Experiment Analysis", fontsize=16, fontweight="bold", y=1.02
        )
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

        plt.close(fig)

    def _print_summary_statistics(self, results_df: pd.DataFrame) -> None:
        """Print summary statistics for the batch experiment results."""
        print("\n=== Benchmark Summary ===")

        heuristic_results = results_df[results_df["solver_type"] == "heuristic"]
        ml_results = results_df[results_df["solver_type"] == "ml"]

        # Heuristic summary
        if not heuristic_results.empty:
            best_heuristic = heuristic_results.loc[
                heuristic_results.groupby("instance_id")["makespan"].idxmin()
            ]
            print("\n--- Best Heuristic Performance ---")
            print(f"Average Makespan: {best_heuristic['makespan'].mean():.2f}")
            print(
                f"Best performing heuristic combination on average: {best_heuristic['solver_name'].mode()[0]}"
            )

        # ML summary
        if not ml_results.empty:
            print("\n--- ML Solvers Performance ---")
            for name, group in ml_results.groupby("solver_name"):
                print(f"\nSolver: {name}")
                print(f"  Average Makespan: {group['makespan'].mean():.2f}")
                if not heuristic_results.empty:
                    avg_improvement = (
                        best_heuristic["makespan"].mean() - group["makespan"].mean()
                    ) / best_heuristic["makespan"].mean()
                    print(
                        f"  Avg. Improvement vs. Best Heuristic: {avg_improvement:.2%}"
                    )

    def plot_results(
        self,
        results_df: pd.DataFrame,
        save_path: Path | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plot solver performance comparison from results DataFrame.

        Handles both ML-only results and mixed heuristic/ML results.
        For ML-only results, shows boxplots comparing different ML solvers.
        For mixed results, shows heuristic baselines (Best, Avg, Worst) alongside ML solvers.
        """
        if results_df.empty:
            print("Cannot plot results, DataFrame is empty.")
            return

        plt.style.use("seaborn-v0_8-whitegrid")

        # Check what types of results we have
        solver_types = results_df["solver_type"].unique()
        has_heuristics = "heuristic" in solver_types
        has_ml = "ml" in solver_types

        heuristic_results = (
            results_df[results_df["solver_type"] == "heuristic"]
            if has_heuristics
            else pd.DataFrame()
        )
        ml_results = (
            results_df[results_df["solver_type"] == "ml"] if has_ml else pd.DataFrame()
        )

        # Build plot data based on available results
        plot_dfs = []
        plot_order = []

        if has_heuristics and not heuristic_results.empty:
            # Get best, average, and worst heuristic for each instance
            best_heuristic_per_instance = heuristic_results.loc[
                heuristic_results.groupby("instance_id")["makespan"].idxmin()
            ].copy()
            best_heuristic_per_instance["solver_name"] = "Best Heuristic"

            avg_heuristic_per_instance = (
                heuristic_results.groupby("instance_id")["makespan"]
                .mean()
                .reset_index()
            )
            avg_heuristic_per_instance["solver_type"] = "heuristic"
            avg_heuristic_per_instance["solver_name"] = "Avg Heuristic"

            worst_heuristic_per_instance = heuristic_results.loc[
                heuristic_results.groupby("instance_id")["makespan"].idxmax()
            ].copy()
            worst_heuristic_per_instance["solver_name"] = "Worst Heuristic"

            plot_dfs.extend(
                [
                    worst_heuristic_per_instance,
                    avg_heuristic_per_instance,
                    best_heuristic_per_instance,
                ]
            )
            plot_order.extend(["Worst Heuristic", "Avg Heuristic", "Best Heuristic"])

        if has_ml and not ml_results.empty:
            ml_solver_names = sorted(ml_results["solver_name"].unique())
            plot_dfs.append(ml_results)
            plot_order.extend(ml_solver_names)

        if not plot_dfs:
            print("No data to plot.")
            return

        plot_df = pd.concat(plot_dfs, ignore_index=True)

        # Create single figure with appropriate width
        num_solvers = len(plot_order)
        fig_width = max(12, num_solvers * 2)
        fig, ax = plt.subplots(figsize=(fig_width, 8))

        # Define colors
        palette = {}
        if has_heuristics:
            palette.update(
                {
                    "Worst Heuristic": "#d62728",  # Red
                    "Avg Heuristic": "#ff7f0e",  # Orange
                    "Best Heuristic": "#2ca02c",  # Green
                }
            )

        # Add distinct colors for each ML solver
        if has_ml and not ml_results.empty:
            ml_solver_names = sorted(ml_results["solver_name"].unique())
            ml_colors = sns.color_palette("Set2", n_colors=len(ml_solver_names))
            for i, ml_name in enumerate(ml_solver_names):
                palette[ml_name] = ml_colors[i]

        # Create boxplot with all solvers
        sns.boxplot(
            data=plot_df,
            x="solver_name",
            y="makespan",
            order=plot_order,
            palette=palette,
            ax=ax,
            linewidth=2,
        )

        # Add stripplot for individual data points
        sns.stripplot(
            data=plot_df,
            x="solver_name",
            y="makespan",
            order=plot_order,
            ax=ax,
            color="black",
            size=5,
            alpha=0.6,
        )

        # Customize plot title based on content
        if has_heuristics and has_ml:
            title = "ML Solver Performance vs. Heuristic Baselines"
        elif has_ml:
            title = "ML Solver Performance Comparison"
        else:
            title = "Heuristic Solver Performance Comparison"

        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Solver", fontsize=13, fontweight="bold")
        ax.set_ylabel("Makespan", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Add median values as text annotations
        medians = plot_df.groupby("solver_name")["makespan"].median()
        for i, solver_name in enumerate(plot_order):
            if solver_name in medians.index:
                median_val = medians[solver_name]
                ax.text(
                    i,
                    median_val,
                    f"{median_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="darkblue",
                )

        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

        plt.close(fig)
