import csv
import sqlite3
from pathlib import Path

import yaml

from jssp_agv.db.schema import MODEL_SCHEMA_SQL, SOLVER_SCHEMA_SQL


class DbSolverClient:
    table_name = "solvers"
    model_name = "models"

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SOLVER_SCHEMA_SQL)
            conn.executescript(MODEL_SCHEMA_SQL)

    def insert_solver(self, row: dict, conn):
        """
        Insert a solver into the solvers table.
        `row` should be a dict containing all necessary columns.
        """
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {self.table_name} (
                solver_name,
                solver_parent,
                instance_generator,
                reward_function,
                job_observation_provider,
                agv_observation_provider,
                total_frames,
                finetune,
                lora,
                env_type,
                number_agvs,
                max_episode_steps,
                seed,
                gnn,
                policy_head,
                rank,
                alpha,
                dropout,
                ml_path,
                log_dir,
                save_dir,
                config_params
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
            """,
            (
                row["solver_name"],
                row["solver_parent"],
                row.get("instance_generator"),
                row.get("reward_function"),
                row.get("job_observation_provider"),
                row.get("agv_observation_provider"),
                row.get("total_frames"),
                int(row.get("finetune", 0)),
                int(row.get("lora", 0)),
                row.get("env_type"),
                row.get("number_agvs"),
                row.get("max_episode_steps"),
                row.get("seed"),
                int(row.get("gnn", 0)),
                int(row.get("policy_head", 0)),
                row.get("rank"),
                row.get("alpha"),
                row.get("dropout"),
                row.get("ml_path"),
                row.get("log_dir"),
                row.get("save_dir"),
                row["config_params"],
            ),
        )

    def exists_solver(self) -> bool:
        """
        Check whether a solver result already exists for a given experiment.
        """
        pass

    def insert_batch_results(self, reader: list[Path]):
        """
        Docstring for insert_batch_results

        :param self: Description
        :param reader: Description
        :type reader: list[Path] List with paths to the models
        """

        with self._connect() as conn:
            for row in reader:
                solver_params = self.get_solver_params_from_path(row)
                self.insert_solver(solver_params, conn)
        conn.close()

    def insert_running_models(self, reader: list[Path]):
        """
        Docstring for insert_running_models

        :param self: Description
        :param reader: Description
        :type reader: list[Path] List with paths to the models
        """

        with self._connect() as conn:
            for row in reader:
                solver_params = self.get_running_models_params(row, conn)
                self.insert_combination(solver_params, conn)
        conn.close()

    def _get_solvers_id_from_path(self, model_path: Path, conn) -> tuple:
        model_path = model_path.as_posix()
        cursor = conn.execute(
            f"""
            SELECT id
            FROM {self.table_name}
            WHERE solver_name = ?
            """,
            (model_path,),
        )

        row = cursor.fetchone()
        if row is None:
            raise KeyError(
                f"Solver not found in DB for path '{model_path}'. "
                f"Expected solver_name='{model_path}'."
            )

        return int(row["id"])

    def get_running_models_params(self, model_path: Path, conn) -> dict:
        if type(model_path) is tuple:
            jssp_model_path, agv_model_path = model_path
            model_type = "decoupled"
        else:
            model_type = "coupled"
            if "ATrue_STrue" in str(model_path):
                model_type = "finetuned"

            jssp_model_path, agv_model_path = model_path, model_path

        jssp_model_solver_id = self._get_solvers_id_from_path(jssp_model_path, conn)
        agv_model_solver_id = self._get_solvers_id_from_path(agv_model_path, conn)
        params = {
            "model_type": model_type,
            "jssp_model_path": jssp_model_path.as_posix(),
            "agv_model_path": agv_model_path.as_posix(),
            "jssp_model_solver_id": jssp_model_solver_id,
            "agv_model_solver_id": agv_model_solver_id,
        }
        print()
        print(f"jssp_id={jssp_model_solver_id} agv_id={agv_model_solver_id}")
        print(f"jssp_model_path={jssp_model_path} agv_model_path={agv_model_path}")
        return params

    def insert_combination(self, row: dict, conn):
        """
        Insert a model into the model table.
        `row` should be a dict containing all necessary columns.
        """
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {self.model_name} (
                model_type,
                jssp_model_path,
                agv_model_path,
                jssp_model_solver_id,
                agv_model_solver_id
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["model_type"],
                row["jssp_model_path"],
                row["agv_model_path"],
                row["jssp_model_solver_id"],
                row["agv_model_solver_id"],
            ),
        )

    def summary(self):
        with self._connect() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) as count FROM {self.table_name}")
            result = cursor.fetchone()
            return result["count"]

    def _classify_solver(self, solver_name: str) -> str:
        """Classify solver based on naming pattern."""
        name_lower = solver_name.lower()
        if "agv" in name_lower:
            return "agv"
        elif "ltrue" in name_lower:
            return "lora"
        elif "fttrue" in name_lower:
            return "ft"
        else:
            return "full"

    def get_solver_stats(self) -> dict:
        """
        Get statistics about solvers in the database.

        Returns:
            dict with solver class counts, distinct rewards, and other stats
        """
        with self._connect() as conn:
            # Get all solvers
            cursor = conn.execute(
                f"SELECT solver_name, reward_function FROM {self.table_name}"
            )
            rows = cursor.fetchall()

            # Classify solvers
            class_counts = {"agv": 0, "lora": 0, "ft": 0, "full": 0}
            for row in rows:
                solver_class = self._classify_solver(row["solver_name"])
                class_counts[solver_class] += 1

            # Get distinct reward functions
            cursor = conn.execute(
                f"SELECT DISTINCT reward_function FROM {self.table_name} WHERE reward_function IS NOT NULL"
            )
            distinct_rewards = [row["reward_function"] for row in cursor.fetchall()]

            return {
                "total_solvers": len(rows),
                "class_counts": class_counts,
                "distinct_rewards": distinct_rewards,
                "num_distinct_rewards": len(distinct_rewards),
            }

    def get_solver_data(self):
        with self._connect() as conn:
            # Get all solvers
            cursor = conn.execute(f"SELECT * FROM {self.table_name}")
            rows = cursor.fetchall()

        configs = {}
        for row in rows:
            config_dict = yaml.safe_load(row["config_params"])
            configs[row["solver_name"]] = config_dict
        return rows, configs

    def plot_solver_overview(self):
        """
        Display an overview of solvers in the database with plots.
        Shows: solver class distribution, reward function distribution.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )
            return

        stats = self.get_solver_stats()

        # Print summary
        print(f"Total solvers: {stats['total_solvers']}")
        print("\nSolvers by class:")
        for cls, count in stats["class_counts"].items():
            print(f"  {cls}: {count}")
        print(f"\nDistinct reward functions ({stats['num_distinct_rewards']}):")
        for reward in stats["distinct_rewards"]:
            print(f"  - {reward}")

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Solver class distribution
        classes = list(stats["class_counts"].keys())
        counts = list(stats["class_counts"].values())
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
        axes[0].bar(classes, counts, color=colors)
        axes[0].set_title("Solvers by Class")
        axes[0].set_xlabel("Solver Class")
        axes[0].set_ylabel("Count")
        for i, (cls, count) in enumerate(zip(classes, counts)):
            axes[0].text(i, count + 0.5, str(count), ha="center", fontweight="bold")

        # Plot 2: Reward function distribution
        with self._connect() as conn:
            cursor = conn.execute(
                f"SELECT reward_function, COUNT(*) as count FROM {self.table_name} "
                f"WHERE reward_function IS NOT NULL GROUP BY reward_function"
            )
            reward_data = cursor.fetchall()

        if reward_data:
            rewards = [row["reward_function"] for row in reward_data]
            reward_counts = [row["count"] for row in reward_data]
            axes[1].barh(rewards, reward_counts, color="#3F51B5")
            axes[1].set_title("Solvers by Reward Function")
            axes[1].set_xlabel("Count")
            for i, count in enumerate(reward_counts):
                axes[1].text(count + 0.2, i, str(count), va="center", fontweight="bold")

        plt.tight_layout()
        plt.show()

        return stats

    def get_solver_params_from_path(self, model_path: Path) -> dict:
        """
        Extract solver parameters from a model path.

        Example:
            model_path = Path("model_repo") / "07-24-44_M_LFalse_FTTrue" / "0"

        Returns a dict suitable for insert_solver.
        """
        # Extract solver_name and solver_parent
        print(f"Inserting solver: {model_path}")
        solver_parent = model_path.parent.name
        solver_name = f"{solver_parent}/{model_path.name}"
        # print(f"  Solver Name: {solver_name}")
        # print(f"  Current Path: {os.getcwd()}")

        # Path to the config.yaml generated by Hydra
        config_path = model_path / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract parameters from config with defaults
        env_cfg = config.get("env", {})
        training_cfg = config.get("training", {})
        finetune_cfg = config.get("finetune", {})

        solver_params = {
            "solver_name": solver_name,
            "solver_parent": solver_parent,
            "instance_generator": env_cfg.get("instance_generator"),
            "reward_function": env_cfg.get("reward_function"),
            "job_observation_provider": env_cfg.get("job_observation_provider"),
            "agv_observation_provider": env_cfg.get("agv_observation_provider"),
            "total_frames": training_cfg.get("total_frames"),
            "finetune": finetune_cfg.get("finetune", False),
            "lora": finetune_cfg.get("lora", False),
            "env_type": config.get("marl", {}).get("env_type"),
            "number_agvs": env_cfg.get("number_agvs"),
            "max_episode_steps": env_cfg.get("max_episode_steps"),
            "seed": config.get("seed"),
            "gnn": finetune_cfg.get("gnn", False),
            "policy_head": finetune_cfg.get("policy_head", False),
            "rank": finetune_cfg.get("rank"),
            "alpha": finetune_cfg.get("alpha"),
            "dropout": finetune_cfg.get("dropout"),
            "ml_path": str(
                finetune_cfg.get("ml_path", model_path)
            ),  # fallback to model path
            "log_dir": str(config.get("log_dir")),
            "save_dir": str(config.get("save_dir")),
            "config_params": yaml.dump(config),
        }

        return solver_params

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """
        Flatten a nested dictionary into a single-level dictionary with dot-separated keys.

        Example:
            {"env": {"reward": "dense"}} -> {"env.reward": "dense"}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def export_to_csv(self, output_path: str = "solver_description.csv") -> Path:
        """
        Export all solver information to a CSV file.

        The config_params YAML is flattened into separate columns with dot-separated keys.

        Args:
            output_path: Path to the output CSV file (default: solver_description.csv)

        Returns:
            Path to the created CSV file
        """
        output_path = Path(output_path)

        with self._connect() as conn:
            cursor = conn.execute(f"SELECT * FROM {self.table_name}")
            rows = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]

        if not rows:
            print("No solvers found in database.")
            return output_path

        # Process each row and flatten config_params
        processed_rows = []
        all_config_keys = set()

        for row in rows:
            row_dict = dict(row)

            # Parse and flatten config_params if present
            config_params_yaml = row_dict.pop("config_params", None)
            flattened_config = {}
            if config_params_yaml:
                try:
                    config = yaml.safe_load(config_params_yaml)
                    if isinstance(config, dict):
                        flattened_config = self._flatten_dict(
                            config, parent_key="config"
                        )
                        all_config_keys.update(flattened_config.keys())
                except yaml.YAMLError as e:
                    print(
                        f"Warning: Could not parse config_params for {row_dict.get('solver_name')}: {e}"
                    )

            # Merge base columns with flattened config
            row_dict.update(flattened_config)
            processed_rows.append(row_dict)

        # Determine final column order: base columns (without config_params) + sorted config keys
        base_columns = [c for c in column_names if c != "config_params"]
        sorted_config_keys = sorted(all_config_keys)
        final_columns = base_columns + sorted_config_keys

        # Write to CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_columns, extrasaction="ignore")
            writer.writeheader()
            for row_dict in processed_rows:
                writer.writerow(row_dict)

        print(f"Exported {len(processed_rows)} solvers to {output_path}")
        print(
            f"Total columns: {len(final_columns)} ({len(base_columns)} base + {len(sorted_config_keys)} config)"
        )

        return output_path


solver_db_client = DbSolverClient("paper_results.db")


def add_all_solver_to_db():
    from jssp_agv.benchmark.solver_collection import (
        AGV_MODEL_PATHS,
        GNN_MODEL_PATHS,
        JSSP_MODEL_PATHS,
        LORA_MODEL_PATHS,
    )

    all_solvers = (
        GNN_MODEL_PATHS
        + LORA_MODEL_PATHS
        + AGV_MODEL_PATHS
        + JSSP_MODEL_PATHS  # + HEURISTIC_SOLVERS
    )
    solver_db_client.insert_batch_results(all_solvers)

    print(f"Wanted to add {len(all_solvers)} solvers to the database.")
    print(f" Summary Count: {solver_db_client.summary()} solvers in database.")


def add_running_models_to_db():
    from itertools import product

    from jssp_agv.benchmark.solver_collection import (
        AGV_MODEL_PATHS,
        GNN_MODEL_PATHS,
        JSSP_MODEL_PATHS,
        LORA_MODEL_PATHS,
    )

    combinations = list(product(JSSP_MODEL_PATHS, AGV_MODEL_PATHS))

    all_solvers = GNN_MODEL_PATHS + LORA_MODEL_PATHS + combinations

    solver_db_client.insert_running_models(all_solvers)


if __name__ == "__main__":
    print(solver_db_client.summary())
    solver_db_client.plot_solver_overview()
    # solver_db_client.export_to_csv()
