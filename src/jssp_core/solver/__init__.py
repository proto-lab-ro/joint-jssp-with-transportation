"""
JSSP Core Solvers

This module contains solver implementations and base classes for solving
Job Shop Scheduling Problems.

Classes:
    SolverType: Enumeration of solver types (heuristic, optimal, ML, etc.)
    JSSPHeuristicSolver: Solver using priority/dispatching rules
    JSSPOptimalSolver: Optimal solver using constraint programming

Functions:
    compare_heuristics: Compare multiple heuristic approaches on an instance
"""

from jssp_core.solver.base import Heuristic, SolverType
from jssp_core.solver.heuristic_solver import JSSPHeuristicSolver


__all__ = [
    "SolverType",
    "JSSPHeuristicSolver",
    "Heuristic",
]
