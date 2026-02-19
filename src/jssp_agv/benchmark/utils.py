from itertools import product
from pathlib import Path

from jssp_agv.benchmark.modules import (
    UniversalSolver,
    _get_agv_scheduler_from_agv_dispatcher,
    _get_agv_scheduler_from_gnn_dispatcher,
    _get_job_scheduler_from_gnn_dispatcher,
    _get_job_scheduler_from_jssp_solver,
)
from jssp_agv.db import DbExperimentsClientBase
from jssp_core.solver.heuristics import agv_heuristic_factory, job_heuristic_factory


# --- Main Execution ---
def get_solver_path(model_path):
    job_solver_path = Path(model_path / "checkpoints" / "policy_module_final.pt")

    if not job_solver_path.exists():
        job_solver_path = Path(model_path / "checkpoints" / "policy_module_final.pt")

        if not job_solver_path.exists():
            raise FileNotFoundError(
                f"Could not find best model at {job_solver_path} or policy_module_final.pt"
            )
    return str(job_solver_path)


def _check_experiment_exists(
    db_client: DbExperimentsClientBase, experiment_params: dict
) -> tuple[bool, int | None]:
    return db_client.exists_experiment(experiment_params)


def get_gnn_solvers(
    ml_solvers: list,
    GNN_MODEL_PATHS: list,
    BATCH_EXPERIMENTS_CONFIG: dict,
    rm,
    db_client: DbExperimentsClientBase,
    experiment_params: dict,
):
    exists_experiment, ex_id = _check_experiment_exists(db_client, experiment_params)
    prev_number_solvers = len(ml_solvers)
    if exists_experiment:
        print("Experiment already exists in the database. Checking for Solver.")

    skipped_solvers = 0
    failed_solvers = 0
    for model_path in GNN_MODEL_PATHS:
        model_dir = Path(model_path)
        if not _check_exisiting_path_w_config(model_dir):
            print(f"Warning: Skipping {model_dir} (not found)")
            skipped_solvers += 1

            continue
        try:
            job_solver_path = get_solver_path(model_path)
            job_solver, job_observatioin_provider, job_observatioin_provider_kwargs = (
                _get_job_scheduler_from_gnn_dispatcher(job_solver_path)
            )

            agv_solver_path = job_solver_path
            agv_solver, agv_observatioin_provider, agv_observatioin_provider_kwargs = (
                _get_agv_scheduler_from_gnn_dispatcher(agv_solver_path)
            )
            solver = UniversalSolver(
                instance=rm,
                job_solver=job_solver,
                agv_solver=agv_solver,
                num_agvs=BATCH_EXPERIMENTS_CONFIG["num_agvs"],
                agv_observation_provider=agv_observatioin_provider,
                agv_observation_kwargs=agv_observatioin_provider_kwargs,
                job_observation_provider=job_observatioin_provider,
                job_observation_kwargs=job_observatioin_provider_kwargs,
            )

            solver_params = {
                "experiment_id": ex_id,
                "solver_name": solver.get_name(),
            }
            if exists_experiment:
                solver_exists = db_client.exists_solver_in_experiment(solver_params)
                if solver_exists:
                    continue

            ml_solvers.append(solver)
            print(f"  Loaded: {solver.get_name()}")
        except Exception as e:
            failed_solvers += 1
            print(f"Warning: Failed to load {model_dir.name}: {e}")
    metrics = {
        "description": "GNN Solvers",
        "totol_solvers": len(GNN_MODEL_PATHS),
        "loaded_solvers": len(ml_solvers) - prev_number_solvers,
        "skipped_solvers": skipped_solvers,
        "failed_solvers": failed_solvers,
    }
    # print(
    #     f"Total GNN Solvers Loaded: {len(ml_solvers) - prev_number_solvers} from {len(GNN_MODEL_PATHS)}"
    # )
    return ml_solvers, metrics


def get_heuristic_solvers(
    HEURISTIC_SOLVERS: list,
    BATCH_EXPERIMENTS_CONFIG: dict,
    rm,
    db_client: DbExperimentsClientBase,
    experiment_params: dict,
):
    exists_experiment, ex_id = _check_experiment_exists(db_client, experiment_params)

    heuristic_solvers = []
    for job_h, agv_h in HEURISTIC_SOLVERS:
        solver = UniversalSolver(
            instance=rm,
            job_solver=job_h,
            agv_solver=agv_h,
            num_agvs=BATCH_EXPERIMENTS_CONFIG["num_agvs"],
            agv_observation_provider=None,
            agv_observation_kwargs={},
            job_observation_provider=None,
            job_observation_kwargs={},
        )

        solver_params = {
            "experiment_id": ex_id,
            "solver_name": solver.get_name(),
        }
        if exists_experiment:
            solver_exists = db_client.exists_solver_in_experiment(solver_params)
            if solver_exists:
                continue
        heuristic_solvers.append(solver)

    metrics = {
        "description": "Heuristic Solvers",
        "totol_solvers": len(HEURISTIC_SOLVERS),
        "loaded_solvers": len(heuristic_solvers),
        "skipped_solvers": len(HEURISTIC_SOLVERS) - len(heuristic_solvers),
        "failed_solvers": 0,
    }
    return heuristic_solvers, metrics


def get_mor_and_scta_heuristic_solvers(
    BATCH_EXPERIMENTS_CONFIG: dict,
    rm,
    db_client: DbExperimentsClientBase,
    experiment_params: dict,
):
    exists_experiment, ex_id = _check_experiment_exists(db_client, experiment_params)

    heuristic_solvers = []

    solver = UniversalSolver(
        instance=rm,
        job_solver=job_heuristic_factory("mor"),
        agv_solver=agv_heuristic_factory("scta"),
        num_agvs=BATCH_EXPERIMENTS_CONFIG["num_agvs"],
        agv_observation_provider=None,
        agv_observation_kwargs={},
        job_observation_provider=None,
        job_observation_kwargs={},
    )

    solver_params = {
        "experiment_id": ex_id,
        "solver_name": solver.get_name(),
    }
    if exists_experiment:
        solver_exists = db_client.exists_solver_in_experiment(solver_params)
        if solver_exists:
            pass
        else:
            heuristic_solvers.append(solver)
    else:
        heuristic_solvers.append(solver)
    # print(
    #     f"Total Heuristic Solvers Loaded: {len(heuristic_solvers)} from {len(HEURISTIC_SOLVERS)}"
    # )
    metrics = {
        "description": "Heuristic Solvers",
        "totol_solvers": 1,
        "loaded_solvers": len(heuristic_solvers),
        "skipped_solvers": 1 - len(heuristic_solvers),
        "failed_solvers": 0,
    }
    return heuristic_solvers, metrics


def get_combined_jssp_and_agv_solvers(
    ml_solvers: list,  # current list of solvers
    JSSP_MODEL_PATHS: list,  # jssp solver paths
    AGV_MODEL_PATHS: list,  # agv solver paths
    BATCH_EXPERIMENTS_CONFIG: dict,
    rm,  # random instance for initialization
    db_client: DbExperimentsClientBase,
    experiment_params: dict,
):
    exists_experiment, ex_id = _check_experiment_exists(db_client, experiment_params)
    prev_number_solvers = len(ml_solvers)
    if exists_experiment:
        print("Experiment already exists in the database. Checking for Solver.")

    combinations = list(product(JSSP_MODEL_PATHS, AGV_MODEL_PATHS))
    # index 0 -> JSSP model path
    # index 1 -> AGV model path
    skipped_solvers = 0
    failed_solvers = 0
    for jssp_model_path, agv_model_path in combinations:
        jssp_model_path_dir = Path(jssp_model_path)
        agv_model_path_dir = Path(agv_model_path)

        if not _check_exisiting_path_w_config(
            jssp_model_path_dir
        ) or not _check_exisiting_path_w_config(agv_model_path_dir):
            print(
                f"Warning: Skipping {jssp_model_path_dir} or {agv_model_path_dir} (not found)"
            )
            skipped_solvers += 1
            continue
        try:
            # ======= DIFFERENT BEFORE

            job_solver_path = get_solver_path(jssp_model_path)
            job_solver, job_observatioin_provider, job_observatioin_provider_kwargs = (
                _get_job_scheduler_from_jssp_solver(job_solver_path)
            )

            agv_solver_path = get_solver_path(agv_model_path)
            agv_solver, agv_observatioin_provider, agv_observatioin_provider_kwargs = (
                _get_agv_scheduler_from_agv_dispatcher(agv_solver_path)
            )
            # ======= SAME AFTERWARDS

            solver = UniversalSolver(
                instance=rm,
                job_solver=job_solver,
                agv_solver=agv_solver,
                num_agvs=BATCH_EXPERIMENTS_CONFIG["num_agvs"],
                agv_observation_provider=agv_observatioin_provider,
                agv_observation_kwargs=agv_observatioin_provider_kwargs,
                job_observation_provider=job_observatioin_provider,
                job_observation_kwargs=job_observatioin_provider_kwargs,
            )

            solver_params = {
                "experiment_id": ex_id,
                "solver_name": solver.get_name(),
            }
            if exists_experiment:
                solver_exists = db_client.exists_solver_in_experiment(solver_params)
                if solver_exists:
                    continue
            ml_solvers.append(solver)
        except Exception as e:
            print(
                f"Warning: Failed to load combination {jssp_model_path_dir} and {agv_model_path_dir}: {e}"
            )
            failed_solvers += 1

    metrics = {
        "description": "Combined JSSP and AGV Solver",
        "totol_solvers": len(combinations),
        "loaded_solvers": len(ml_solvers) - prev_number_solvers,
        "skipped_solvers": skipped_solvers,
        "failed_solvers": failed_solvers,
    }

    return ml_solvers, metrics


def _check_exisiting_path_w_config(path: Path) -> bool:
    return path.exists() and (path / ".hydra" / "config.yaml").exists()


def summarize_solver_loading_metrics(
    metrics_list: list[dict], print_summary: bool = True
) -> dict:
    """
    Summarize solver loading metrics and optionally display a formatted table.

    Args:
        metrics_list: List of metric dictionaries from solver loading functions.
        print_summary: Whether to print the formatted summary table (default: True).

    Returns:
        Dictionary containing the aggregated summary statistics.
    """
    summary = {}
    grand_total = {"total": 0, "loaded": 0, "skipped": 0, "failed": 0}

    for metrics in metrics_list:
        desc = metrics["description"]
        # Handle both "totol_solvers" (legacy typo) and "total_solvers"
        total = metrics.get("total_solvers", metrics.get("totol_solvers", 0))
        loaded = metrics.get("loaded_solvers", 0)
        skipped = metrics.get("skipped_solvers", 0)
        failed = metrics.get("failed_solvers", 0)

        summary[desc] = {
            "total_solvers": total,
            "loaded_solvers": loaded,
            "skipped_solvers": skipped,
            "failed_solvers": failed,
            "success_rate": (loaded / total * 100) if total > 0 else 0.0,
        }

        grand_total["total"] += total
        grand_total["loaded"] += loaded
        grand_total["skipped"] += skipped
        grand_total["failed"] += failed

    # Calculate overall success rate
    grand_total["success_rate"] = (
        (grand_total["loaded"] / grand_total["total"] * 100)
        if grand_total["total"] > 0
        else 0.0
    )
    summary["_grand_total"] = grand_total

    if print_summary:
        _print_solver_loading_summary(summary)

    return summary


def _print_solver_loading_summary(summary: dict) -> None:
    """Print a professionally formatted solver loading summary table."""
    # Determine dynamic category width based on longest category name
    max_category_len = max(len(cat) for cat in summary.keys() if cat != "_grand_total")
    # Add 2 for status indicator and space, minimum 20, cap at 40
    category_width = min(max(max_category_len + 4, 20), 40)

    # Table configuration
    col_widths = {
        "category": category_width,
        "total": 7,
        "loaded": 8,
        "skipped": 8,
        "failed": 8,
        "rate": 9,
    }
    total_width = sum(col_widths.values()) + 8  # 8 for separators

    # Header
    print("\n" + "=" * total_width)
    print("SOLVER LOADING SUMMARY".center(total_width))
    print("=" * total_width)

    # Column headers
    header = (
        f"│ {'Category':<{col_widths['category']}} "
        f"│ {'Total':>{col_widths['total']}} "
        f"│ {'Loaded':>{col_widths['loaded']}} "
        f"│ {'Skipped':>{col_widths['skipped']}} "
        f"│ {'Failed':>{col_widths['failed']}} "
        f"│ {'Rate':>{col_widths['rate']}} │"
    )
    print(header)
    separator = (
        "├"
        + "─" * (col_widths["category"] + 2)
        + "┼"
        + "─" * (col_widths["total"] + 2)
        + "┼"
        + "─" * (col_widths["loaded"] + 2)
        + "┼"
        + "─" * (col_widths["skipped"] + 2)
        + "┼"
        + "─" * (col_widths["failed"] + 2)
        + "┼"
        + "─" * (col_widths["rate"] + 2)
        + "┤"
    )
    print(separator)

    # Data rows (excluding grand total)
    for category, stats in summary.items():
        if category == "_grand_total":
            continue

        total = stats["total_solvers"]
        loaded = stats["loaded_solvers"]
        skipped = stats["skipped_solvers"]
        failed = stats["failed_solvers"]
        rate = stats["success_rate"]

        # Status indicator
        if rate == 100:
            status = "✓"
        elif rate == 0 and total > 0:
            status = "✗"
        elif failed > 0:
            status = "!"
        elif skipped > 0:
            status = "○"
        else:
            status = " "

        # Truncate category if needed (leave room for status + space + ellipsis)
        display_cat = category
        max_cat_display = col_widths["category"] - 2  # account for status + space
        if len(display_cat) > max_cat_display:
            display_cat = display_cat[: max_cat_display - 1] + "…"

        row = (
            f"│ {status} {display_cat:<{col_widths['category'] - 2}} "
            f"│ {total:>{col_widths['total']}} "
            f"│ {loaded:>{col_widths['loaded']}} "
            f"│ {skipped:>{col_widths['skipped']}} "
            f"│ {failed:>{col_widths['failed']}} "
            f"│ {rate:>{col_widths['rate'] - 1}.1f}% │"
        )
        print(row)

    # Separator before grand total
    print(separator)

    # Grand total row
    gt = summary["_grand_total"]
    grand_row = (
        f"│ {'TOTAL':<{col_widths['category']}} "
        f"│ {gt['total']:>{col_widths['total']}} "
        f"│ {gt['loaded']:>{col_widths['loaded']}} "
        f"│ {gt['skipped']:>{col_widths['skipped']}} "
        f"│ {gt['failed']:>{col_widths['failed']}} "
        f"│ {gt['success_rate']:>{col_widths['rate'] - 1}.1f}% │"
    )
    print(grand_row)
    print("=" * total_width + "\n")


# Keep the old function name as an alias for backwards compatibility
summaryize_solver_loading_metrics = summarize_solver_loading_metrics
