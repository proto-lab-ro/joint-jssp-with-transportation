"""
Observation providers for JSSP environments.

This module provides a flexible system for creating different observation formats
without modifying the core environment logic. Different observation providers can
be easily switched to experiment with various state representations.
"""

import inspect
import logging

import numpy as np
import torch
from gymnasium import spaces
from torchrl.data import (
    NonTensor,
)
from torchrl.envs.libs.gym import convert_dict_spec

from jssp_core.domain import GraphNormalization, ObservationType
from jssp_core.domain.observation import ObservationData, ObservationProvider
from jssp_core.registry import OBSERVATION_REGISTRY
from jssp_core.schedule import Schedule


logger = logging.getLogger(__name__)


@OBSERVATION_REGISTRY.register("lb_bipartite_gnn")
class LbGnnBipartiteObservationProvider(ObservationProvider):
    """
    GNN observation provider for the Operation–Machine Bipartite Graph.

    Assuming that after initialization, the schedule instance will not change size.

    Each node represents either:
      - an operation (job–machine pair), or
      - a machine (resource node).

    Edges connect:
      - successive operations in the same job (precedence edges)
      - each operation to its assigned machine (resource edges)

    Output:
      node_feats: [num_ops + num_machines, feat_dim]
      edge_index: [2, num_edges]

    Args:
        schedule: The schedule instance.
        max_nodes: Maximum number of nodes (for padding).
        include_precedence_edges: Whether to include precedence edges.
        both_directions: Whether to include edges in both directions.
        self_loop: Whether to include self-loops.
        observation_type: The type of observation to return.
        normalize: Normalization strategy. Can be a float for manual scaling (divides by this value),
                   or a GraphNormalization enum for automatic strategies (JOB, OPERATION).
                   Defaults to GraphNormalization.NONE (scale=1.0).
    """

    def __init__(
        self,
        schedule: Schedule,
        max_nodes: int | None = None,
        include_precedence_edges: bool = True,
        both_directions: bool = False,
        self_loop: bool = False,
        observation_type: ObservationType = ObservationType.GRAPH,
        normalize: float | GraphNormalization = GraphNormalization.NONE,
    ):
        super().__init__(schedule)

        # --- (1) Base dimensions ---------------------------------------------------
        # --- Varying per instance ---
        self.num_machines = schedule.num_machines
        self.num_ops = schedule.num_operations
        self.max_nodes = max_nodes or (self.num_ops + self.num_machines)

        # --- Constants
        self.include_precedence_edges = include_precedence_edges
        self.both_directions = both_directions
        self.self_loop = self_loop
        self._observation_type = observation_type

        # --- Varying per instance ---
        self.max_edges = self._estimate_max_edges(schedule)

        # Handle normalization configuration
        if isinstance(normalize, (int, float)):
            self.normalize = GraphNormalization.NONE
            self.scale = float(normalize)
        else:
            self.normalize = normalize
            self.scale = 1.0

        if self._observation_type not in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.GRAPH} and {ObservationType.GRAPH_MATRIX} are allowed."
            )

        self.num_node_features = 3


        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1
        self.machine_offset = node_id  

    def _estimate_max_edges(self, schedule: Schedule) -> int:
        """
        Pre-calculate max_edges once since all instances share the same size.
        """
        precedence_edges = 0
        if self.include_precedence_edges:
            precedence_edges = sum(max(len(job) - 1, 0) for job in schedule.instance)
            if self.both_directions:
                precedence_edges *= 2

        op_machine_edges = self.num_ops * 2  
        self_loop_edges = self.num_ops + self.num_machines if self.self_loop else 0

        return precedence_edges + op_machine_edges + self_loop_edges

    @property
    def name(self) -> str:
        return "lb_bipartite_gnn"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def reset(self, schedule: Schedule) -> None:
        super().reset(schedule)
        self.num_machines = schedule.num_machines
        self.num_ops = schedule.num_operations
        self.max_nodes = self.num_ops + self.num_machines
        self.max_edges = self._estimate_max_edges(schedule)
        self.scale = self.schedule.get_lower_bound_makespan()

        # Create operation to node mapping
        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1
        self.machine_offset = node_id  

    def get_observation_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "node_feats": spaces.Box(
                    -np.inf,
                    np.inf,
                    (self.max_nodes, self.num_node_features),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    0, self.max_nodes - 1, (2, self.max_edges), dtype=np.int64
                ),
            }
        )

    def get_observation(self, schedule: Schedule) -> ObservationData:
        node_feats = self._build_node_features(schedule)
        edge_index = self._build_edge_index(schedule)
        obs: ObservationData = {"node_feats": node_feats, "edge_index": edge_index}
        return obs

    def _build_node_features(self, schedule: Schedule) -> np.ndarray:
        """
        Build features for all operation + machine nodes.

        Shared schema:
            [0] normalized_lower_bound (ops) / machine_progress (machines)
            [1] is_scheduled (ops) / 0 (machines)
            [2] is_machine flag (0=op, 1=machine)
        """
        num_nodes = self.num_ops + self.num_machines
        feats = np.zeros((num_nodes, self.num_node_features), dtype=np.float32)

        # ---- (A) Operation nodes -------------------------------------------------
        scheduled_ops = schedule.scheduled
        lower_bounds = schedule.get_operation_lower_bounds()
        if self.normalize == GraphNormalization.JOB:
            max_lb_per_job = self._get_max_lower_bound_per_job(lower_bounds, schedule)

        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine_id, _) in enumerate(job):
                node = self.op_node_id[(job_idx, op_idx)]

                lb_norm = lower_bounds[(job_idx, op_idx)]
                if self.normalize == GraphNormalization.JOB:
                    lb_norm /= (
                        max_lb_per_job[job_idx] if max_lb_per_job[job_idx] > 0 else 1.0
                    )
                elif self.normalize == GraphNormalization.OPERATION:
                    lb_norm = lb_norm
                else:
                    lb_norm /= self.scale
                feats[node, 0] = lb_norm
                feats[node, 1] = float((job_idx, op_idx) in scheduled_ops)
                feats[node, 2] = 0.0  
        if self.normalize == GraphNormalization.OPERATION:
            feats[:, 0] = self._normalize_by_all_operations(feats[:, 0])

        # ---- (B) Machine nodes ---------------------------------------------------
        for m_id in range(self.num_machines):
            node = self.machine_offset + m_id
            ops_for_machine = schedule.get_operations_on_machine(m_id)
            if len(ops_for_machine) > 0:
                num_sched = sum(1 for op in ops_for_machine if op in scheduled_ops)
                progress = num_sched / len(ops_for_machine)
            else:
                progress = 0.0

            feats[node, 0] = progress  # machine "load progress"
            feats[node, 1] = 0.0  # unused in constraint-based env
            feats[node, 2] = 1.0  # is_machine flag

        return feats

    def _normalize_by_all_operations(self, feature: np.ndarray) -> np.ndarray:
        max_value = np.max(feature) if np.max(feature) > 0 else 1.0
        min_value = np.min(feature) if np.min(feature) < 0 else -1.0
        normalized_feature = (feature - min_value) / (max_value - min_value)
        return normalized_feature

    def _get_max_lower_bound_per_job(
        self, lower_bounds: dict[tuple[int, int], float], schedule: Schedule
    ) -> dict[int, float]:
        max_lb_per_job = {}
        for job_idx, job in enumerate(schedule.instance):
            job_lbs = [
                lower_bounds.get((job_idx, op_idx), 0.0) for op_idx in range(len(job))
            ]
            max_lb_per_job[job_idx] = max(job_lbs) if job_lbs else 0.0
        return max_lb_per_job

    def _build_edge_index(self, schedule: Schedule) -> np.ndarray:
        """
        Build edge index for precedence and operation–machine edges.
        Returns (2, num_edges)
        """
        edges = []

        # ---- (A) Precedence edges ----------------------------------------------
        if self.include_precedence_edges:
            for job_idx, job in enumerate(schedule.instance):
                for op_idx in range(1, len(job)):
                    src = self.op_node_id[(job_idx, op_idx - 1)]
                    dst = self.op_node_id[(job_idx, op_idx)]
                    edges.append((src, dst))
                    if self.both_directions:
                        edges.append((dst, src))

        # ---- (B) Operation–Machine edges ----------------------------------------
        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine_id, _) in enumerate(job):
                op_node = self.op_node_id[(job_idx, op_idx)]
                mach_node = self.machine_offset + machine_id
                edges.append((op_node, mach_node))
                edges.append((mach_node, op_node))

        # ---- (C) Self-loops -----------------------------------------------------
        if self.self_loop:
            for n in range(self.num_ops + self.num_machines):
                edges.append((n, n))

        edge_index = np.array(edges, dtype=np.int64).T  

        if edge_index.shape[1] > self.max_edges:
            raise ValueError(
                f"Number of edges {edge_index.shape[1]} exceeds max_edges {self.max_edges}"
            )

        return edge_index

    def get_observation_space_trl(self):
        if self.observation_type == ObservationType.GRAPH:
            return NonTensor(
                shape=torch.Size(
                    [
                        1,
                    ]
                )
            )

        elif self.observation_type == ObservationType.GRAPH_MATRIX:
            return convert_dict_spec(self.get_observation_space())



OBSERVATION_PROVIDERS = OBSERVATION_REGISTRY._registry


def get_observation_provider(
    name: str, schedule: Schedule, **kwargs
) -> ObservationProvider:
    """
    Factory function to create observation providers.

    Args:
        name: Name of the observation provider.
        schedule: Schedule instance for initialization.
        **kwargs: Additional arguments for specific providers.

    Returns:
        ObservationProvider instance.

    Raises:
        ValueError: If provider name is not recognized.
    """
    provider_class = OBSERVATION_REGISTRY.get(name)

   
    sig = inspect.signature(provider_class.__init__)
    params = sig.parameters

    # Does the __init__ accept **kwargs?
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if has_var_kw:
        return provider_class(schedule, **kwargs)

    accepted_names = {
        name
        for name, p in params.items()
        if name not in ("self", "schedule")
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_names}
    unused_kwargs = {k: v for k, v in kwargs.items() if k not in accepted_names}

    if unused_kwargs:
        logger.warning(
            "Unused keyword arguments for provider '%s' in factory: %s",
            name,
            ", ".join(sorted(unused_kwargs.keys())),
        )

    return provider_class(schedule, **filtered_kwargs)
