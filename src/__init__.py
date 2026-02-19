"""
JSSP GNN Baseline

A modular framework for Job Shop Scheduling Problem (JSSP) research with
Graph Neural Network support and Reinforcement Learning capabilities.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main package imports for convenience
from jssp_core import Schedule
from jssp_core.environments.jssp import JSSPEnv
from jssp_core.instances import get_instance
from jssp_core.solver.heuristic_solver import JSSPHeuristicSolver


__all__ = [
    "Schedule",
    "JSSPEnv",
    "JSSPHeuristicSolver",
    # Factories
    "get_instance",
]
