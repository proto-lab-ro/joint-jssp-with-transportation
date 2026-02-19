import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from .schema import BM_SCHEMA_SQL, GRID_SCHEMA_SQL


class DbExperimentsClientBase(ABC):
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @abstractmethod
    def _init_db(self):
        pass

    @abstractmethod
    def insert_experiment_if_unique(self, params: dict) -> int:
        pass

    @abstractmethod
    def insert_result(self, experiment_id: int, row: dict, conn):
        pass

    def insert_batch_results(self, reader, experiment_id: int):
        with self._connect() as conn:
            for row in reader:
                self.insert_result(experiment_id, row, conn)
        conn.close()

    @abstractmethod
    def exists_experiment(self, params: dict) -> tuple[bool, int | None]:
        """
        Check whether an experiment with exactly these parameters exists.

        Returns:
            (exists: bool, experiment_id: int | None)
        """
        pass

    @abstractmethod
    def exists_solver_in_experiment(self, solver_params: dict) -> bool:
        """
        Check whether a solver result already exists for a given experiment.
        """
        pass

    def normalize_solver_name(self, ml_path: str, ml: bool) -> str:
        """
        Convert a path like:
        2025-12-13\08-40-46_M_LFalse_FTTrue\1\checkpoints\best_model.pt
        to:
        08-40-46_M_LFalse_FTTrue/1
        """
        if not ml:
            return ml_path  # Heuristic solvers remain unchanged
        p = Path(ml_path)
        # print(p)
        # Folder name containing M_LFalse_FTTrue
        solver_parent = p.parents[2].name  # parent of "1/checkpoints/best_model.pt"
        instance_id = p.parents[1].name  # numeric folder, e.g. "1"
        return f"{solver_parent}/{instance_id}"


class GridDatabaseClient(DbExperimentsClientBase):
    def __init__(self, db_path: str):
        super().__init__(db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(GRID_SCHEMA_SQL)

    def insert_experiment_if_unique(self, params: dict) -> int:
        query = """
        INSERT INTO grid_experiments (
            num_instances, num_jobs, num_machines, num_agvs,
            max_duration, number_cells, seed_start,
            interval, std, grid_type 
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(
            num_instances, num_jobs, num_machines, num_agvs,
            max_duration, number_cells, interval, std,
            grid_type, seed_start
        ) DO NOTHING
        RETURNING id;
        """

        values = (
            params["num_instances"],
            params["num_jobs"],
            params["num_machines"],
            params["num_agvs"],
            params["max_duration"],
            params["number_cells"],
            params["seed_start"],
            params["interval"],
            params["std"],
            params["grid_type"],
        )

        with self._connect() as conn:
            cur = conn.execute(query, values)
            row = cur.fetchone()
            if row is not None:
                # New row inserted
                return row["id"]
            else:
                # Row already exists, fetch the existing id
                check_query = """
                SELECT id FROM grid_experiments
                WHERE num_instances=? AND num_jobs=? AND num_machines=? AND num_agvs=?
                AND max_duration=? AND number_cells=? AND seed_start=? AND interval=? AND std=? AND grid_type=?
                """
                cur = conn.execute(check_query, values)
                row = cur.fetchone()
                if row is not None:
                    return row["id"]
                else:
                    raise ValueError("Failed to insert or find experiment ID")

    def insert_result(self, experiment_id: int, row: dict, conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO grid_results (
                grid_experiments_id, instance_id,
                agv_solver_type, job_solver_type, solver_type,
                solver_name, makespan, job_model, agv_model, seed,
                nr_inst, min_travel, min_op, job_model_name, agv_model_name, op2transport_ratio 
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                row["instance_id"],
                row["agv_solver_type"],
                row["job_solver_type"],
                row["solver_type"],
                row["solver_name"],
                row["makespan"],
                row["job_model"],
                row["agv_model"],
                row["seed"],
                row["nr_inst"],
                row["min_travel"],
                row["min_op"],
                self.normalize_solver_name(
                    row["job_model"], row["solver_type"] == "ml"
                ),
                self.normalize_solver_name(
                    row["agv_model"], row["solver_type"] == "ml"
                ),
                row["op2transport_ratio"],
            ),
        )

    def exists_experiment(self, params: dict) -> tuple[bool, int | None]:
        query = """
        SELECT id
        FROM grid_experiments
        WHERE num_instances=? AND num_jobs=? AND num_machines=? AND num_agvs=?
          AND max_duration=? AND number_cells=? AND seed_start=?
          AND interval=? AND std=? AND grid_type=?
        LIMIT 1;
        """

        values = (
            params["num_instances"],
            params["num_jobs"],
            params["num_machines"],
            params["num_agvs"],
            params["max_duration"],
            params["number_cells"],
            params["seed_start"],
            params["interval"],
            params["std"],
            params["grid_type"],
        )

        with self._connect() as conn:
            row = conn.execute(query, values).fetchone()

        if row:
            return True, row["id"]
        return False, None

    def exists_solver_in_experiment(self, solver_params: dict) -> bool:
        query = """
        SELECT 1
        FROM grid_results
        WHERE grid_experiments_id = ?
        AND solver_name = ?
        LIMIT 1;
        """

        values = (
            solver_params["experiment_id"],
            solver_params["solver_name"],
        )

        with self._connect() as conn:
            row = conn.execute(query, values).fetchone()

        return row is not None


class BMDatabaseClient(DbExperimentsClientBase):
    def __init__(self, db_path: str):
        super().__init__(db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(BM_SCHEMA_SQL)

    def insert_experiment_if_unique(self, params: dict) -> int:
        query = """
        INSERT INTO bm_experiments (
            num_instances, num_jobs, num_machines, num_agvs,
            min_operation_duration, max_operation_duration,
            min_travel_time, max_travel_time, seed_start
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(
            num_instances, num_jobs, num_machines, num_agvs,
            min_operation_duration, max_operation_duration,
            min_travel_time, max_travel_time, seed_start
        ) DO NOTHING
        RETURNING id;
        """

        values = (
            params["num_instances"],
            params["num_jobs"],
            params["num_machines"],
            params["num_agvs"],
            params["min_operation_duration"],
            params["max_operation_duration"],
            params["min_travel_time"],
            params["max_travel_time"],
            params["seed_start"],
        )

        with self._connect() as conn:
            cur = conn.execute(query, values)
            row = cur.fetchone()
            if row is not None:
                # New row inserted
                return row["id"]
            else:
                # Row already exists, fetch the existing id
                check_query = """
                SELECT id FROM bm_experiments
                WHERE num_instances=? AND num_jobs=? AND num_machines=? AND num_agvs=?
                AND min_operation_duration=? AND max_operation_duration=? 
                AND min_travel_time=? AND max_travel_time=? AND seed_start=?
                """
                cur = conn.execute(check_query, values)
                row = cur.fetchone()
                if row is not None:
                    return row["id"]
                else:
                    raise ValueError("Failed to insert or find experiment ID")

    def insert_result(self, experiment_id: int, row: dict, conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO bm_results (
                bm_experiments_id, instance_id,
                agv_solver_type, job_solver_type, solver_type,
                solver_name, makespan, job_model, agv_model, seed, job_model_name, agv_model_name
            )
            VALUES (?,  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                row["instance_id"],
                row["agv_solver_type"],
                row["job_solver_type"],
                row["solver_type"],
                row["solver_name"],
                row["makespan"],
                row["job_model"],
                row["agv_model"],
                row["seed"],
                self.normalize_solver_name(
                    row["job_model"], row["solver_type"] == "ml"
                ),
                self.normalize_solver_name(
                    row["agv_model"], row["solver_type"] == "ml"
                ),
            ),
        )

    def exists_experiment(self, params: dict) -> tuple[bool, int | None]:
        query = """
        SELECT id
        FROM bm_experiments
        WHERE num_instances=? AND num_jobs=? AND num_machines=? AND num_agvs=?
          AND min_operation_duration=? AND max_operation_duration=?
          AND min_travel_time=? AND max_travel_time=? AND seed_start=?
        LIMIT 1;
        """

        values = (
            params["num_instances"],
            params["num_jobs"],
            params["num_machines"],
            params["num_agvs"],
            params["min_operation_duration"],
            params["max_operation_duration"],
            params["min_travel_time"],
            params["max_travel_time"],
            params["seed_start"],
        )

        with self._connect() as conn:
            row = conn.execute(query, values).fetchone()

        if row:
            return True, row["id"]
        return False, None

    def exists_solver_in_experiment(self, solver_params: dict) -> bool:
        query = """
        SELECT 1
        FROM bm_results
        WHERE bm_experiments_id = ?
        AND solver_name = ?
        LIMIT 1;
        """

        values = (
            solver_params["experiment_id"],
            solver_params["solver_name"],
        )

        with self._connect() as conn:
            row = conn.execute(query, values).fetchone()

        return row is not None


grid_db_client = GridDatabaseClient("paper_results.db")
bm_db_client = BMDatabaseClient("paper_results.db")
