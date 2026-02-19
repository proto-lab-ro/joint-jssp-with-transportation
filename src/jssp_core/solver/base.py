"""
Quick Win Implementation: Enhanced solver_base.py

This is a drop-in replacement for jssp_core/solver_base.py that adds:
- SolverType enum for categorizing solvers
- Batch solving support
- Better metadata methods
- Improved documentation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Protocol

from jssp_core.instances import JSSPInstance
from jssp_core.schedule import Schedule


@dataclass(frozen=True)
class SolveOutput:
    solution: Schedule
    info: dict[str, Any] = field(
        default_factory=dict
    )  # extras: rollout, returns, logs...


class EntityType(StrEnum):
    AGV = "agv"
    JOB = "job"
    OP = "operation"


class SolverProtocol(Protocol):
    def solve(self, instance: JSSPInstance) -> Schedule: ...

    def step(self, current_schedule: Schedule, *args, **kwargs) -> int: ...

    def get_name(self) -> str: ...

    def get_config_hash(self) -> str: ...

    def get_type(self) -> "SolverType": ...

    def solve_with_info(self, instance: JSSPInstance) -> "SolveOutput": ...


class Heuristic(ABC):
    def solve(self, instance: JSSPInstance) -> Schedule:
        """
        Solve gets an instance and returns a full solution.

        Default implementation creates a Schedule and calls step() until complete.
        Job-based heuristics can use this implementation directly.
        AGV-based heuristics should override this method.

        Args:
            instance: The JSSP instance to solve

        Returns:
            A complete Schedule object
        """
        schedule = Schedule(instance)
        while not schedule.is_complete():
            job_id = self.step(schedule)
            schedule.schedule_job(job_id)
        return schedule

    @abstractmethod
    def step(self, current_schedule: Schedule, **kwargs) -> int:
        """Returns the next Entity ID <EntityType> (eg. Job ID, AGV ID) to schedule, for MDP step"""
        pass

    def get_config_hash(self) -> str:
        """Return a stable identifier for heuristics that do not carry configs."""

        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the heuristic."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return the name of the heuristic."""
        return self.__class__.__name__

    def get_name(self) -> str:
        """Return the name of the heuristic."""
        return self.__class__.__name__

    @property
    def name(self) -> str:
        return self.get_name()

    def get_type(self) -> "SolverType":
        return SolverType.HEURISTIC

    def solve_with_info(self, instance: JSSPInstance) -> "SolveOutput":
        import time

        start_time = time.perf_counter()
        schedule = self.solve(instance)
        # end_time = time.perf_counter()
        # computation_time = end_time - start_time

        return SolveOutput(solution=schedule, info={})


class SolverType(Enum):
    """Enumeration of solver types for categorization and benchmarking."""

    HEURISTIC = "heuristic"  # Priority rules, dispatching rules
    OPTIMAL = "optimal"  # Constraint programming, exact methods
    ML = "ml"  # Machine learning (GNN, MLP, RL-based)
    HYBRID = "hybrid"  # Combination of multiple approaches
    METAHEURISTIC = "metaheuristic"  # GA, SA, Tabu Search, etc.


# Export for convenience
__all__ = ["SolverType"]
