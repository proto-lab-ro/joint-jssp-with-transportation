"""
Universal Solver for JSSP with AGV instances.

This module provides a unified solver that can work with both heuristic-based
and ML-based approaches for Job Shop Scheduling Problems with AGVs.

Classes:
    UniversalSolver: A universal solver that accepts either heuristic or ML
                     solvers for both Job and AGV scheduling.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jssp_agv.dispatcher import (
    AgvDispatcher,
    GNNDispatcher,
    JsspDispatcher,
)
from jssp_agv.dispatcher.utils import lora_params
from jssp_agv.train_gnn_jssp import _get_agv_selector
from jssp_agv.utils.load_model import load_policy_config
from jssp_core.domain.domains import ItemDataType
from jssp_core.instances.agv import JSSPTransportInstance
from jssp_core.observation_providers import (
    ObservationProvider,
    get_observation_provider,
)
from jssp_core.observation_providers.agv import get_agv_observation_provider
from jssp_core.schedule import TransportSchedule
from jssp_core.solver.base import Heuristic
from jssp_core.solver.heuristics import agv_heuristic_factory, job_heuristic_factory


class UniversalSolver:
    """
    Universal solver for JSSP Transport instances with AGVs.

    This solver can work with any combination of:
    - Job scheduler: Heuristic or ML (torch.nn.Module)
    - AGV scheduler: Heuristic or ML (torch.nn.Module)

    The solver automatically detects whether each scheduler is a heuristic
    or ML model and handles observation provider initialization accordingly.

    Example:
        # Heuristic + Heuristic
        solver = UniversalSolver(
            instance=instance,
            job_solver=SPT(),
            agv_solver=SCPT(),
            num_agvs=3
        )

        # ML + ML
        solver = UniversalSolver(
            instance=instance,
            job_solver=job_policy_module,
            agv_solver=agv_policy_module,
            job_observation_provider="default",
            job_observation_kwargs={},
            agv_observation_provider="default",
            agv_observation_kwargs={},
            job_selector_kwargs={"job_selector_type": ItemDataType.JOB},
            agv_selector_kwargs={},
            num_agvs=3
        )

        # Heuristic + ML (mixed)
        solver = UniversalSolver(
            instance=instance,
            job_solver=SPT(),
            agv_solver=agv_policy_module,
            agv_observation_provider="default",
            agv_observation_kwargs={},
            num_agvs=3
        )

        # Solve the instance
        makespan, schedule = solver.solve()
    """

    def __init__(
        self,
        instance: JSSPTransportInstance,
        job_solver: Heuristic | torch.nn.Module,
        agv_solver: Heuristic | torch.nn.Module,
        num_agvs: int = 3,
        job_observation_provider: str | ObservationProvider | None = None,
        job_observation_kwargs: dict[str, Any] | None = None,
        agv_observation_provider: str | ObservationProvider | None = None,
        agv_observation_kwargs: dict[str, Any] | None = None,
        job_selector_kwargs: dict[str, Any] | None = None,
        agv_selector_kwargs: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
    ):
        """
        Initialize the UniversalSolver.

        Parameters
        ----------
        instance : JSSPTransportInstance
            The JSSP transport instance to solve
        job_solver : Heuristic | torch.nn.Module
            Either a job heuristic or ML policy module
        agv_solver : Heuristic | torch.nn.Module
            Either an AGV heuristic or ML policy module
        num_agvs : int
            Number of AGVs in the system (default: 3)
        job_observation_provider : Optional[str | ObservationProvider]
            Observation provider for job scheduler (required if job_solver is ML)
        job_observation_kwargs : Optional[dict]
            Keyword arguments for job observation provider
        agv_observation_provider : Optional[str | ObservationProvider]
            Observation provider for AGV scheduler (required if agv_solver is ML)
        agv_observation_kwargs : Optional[dict]
            Keyword arguments for AGV observation provider
        job_selector_kwargs : Optional[dict]
            Keyword arguments for job selector (e.g., job_selector_type)
        agv_selector_kwargs : Optional[dict]
            Keyword arguments for AGV selector
        device : str | torch.device
            Device for ML inference (default: "cpu")
        """
        self.instance = instance
        self.job_solver = job_solver
        self.agv_solver = agv_solver
        self.num_agvs = num_agvs
        self.device = torch.device(device)

        # Store selector kwargs with defaults
        self.job_selector_kwargs = job_selector_kwargs or {}
        self.agv_selector_kwargs = agv_selector_kwargs or {}

        # Set default job_selector_type if not provided
        if "job_selector_type" not in self.job_selector_kwargs:
            self.job_selector_kwargs["job_selector_type"] = ItemDataType.JOB

        # Track selected job for AGV selector
        self.action_selected_job_id = None

        # Create TransportSchedule from the instance
        self.schedule = TransportSchedule(instance, num_agvs=num_agvs)

        # Determine solver types
        self.job_is_ml = isinstance(job_solver, torch.nn.Module)
        self.agv_is_ml = isinstance(agv_solver, torch.nn.Module)
        self.job_is_heuristic = isinstance(job_solver, Heuristic)
        self.agv_is_heuristic = isinstance(agv_solver, Heuristic)

        # Initialize observation providers for ML solvers
        self.job_observation_provider = None
        self.agv_observation_provider = None
        self.job_observation_kwargs = job_observation_kwargs or {}
        self.agv_observation_kwargs = agv_observation_kwargs or {}
        if self.job_is_ml:
            if job_observation_provider is None:
                raise ValueError(
                    "job_observation_provider must be provided when job_solver is an ML model"
                )

            if isinstance(job_observation_provider, str):
                self.job_observation_provider = get_observation_provider(
                    job_observation_provider,
                    self.schedule,
                    **self.job_observation_kwargs,
                )
            else:
                self.job_observation_provider = job_observation_provider

            # Move job solver to device and set to eval mode
            self.job_solver.to(self.device)
            self.job_solver.eval()

        if self.agv_is_ml:
            if agv_observation_provider is None:
                raise ValueError(
                    "agv_observation_provider must be provided when agv_solver is an ML model"
                )

            if isinstance(agv_observation_provider, str):
                self.agv_observation_provider = get_agv_observation_provider(
                    agv_observation_provider,
                    self.schedule,
                    **self.agv_observation_kwargs,
                )
            else:
                self.agv_observation_provider = agv_observation_provider

            # Move agv solver to device and set to eval mode
            self.agv_solver.to(self.device)
            self.agv_solver.eval()

    def _get_solver_name(
        self, solver: Heuristic | torch.nn.Module, kwargs: dict
    ) -> str:
        """Get a human-readable name for a solver."""
        if isinstance(solver, Heuristic):
            return solver.__class__.__name__
        elif isinstance(solver, torch.nn.Module):
            model_path = kwargs.get("model_path", None)
            model_path = model_path.replace("\\", "/") if model_path else None
            model_path = model_path.replace("outputs", "") if model_path else None
            model_path = model_path.replace("multirun", "") if model_path else None
            return f"{model_path})"
        else:
            return str(type(solver).__name__)

    def get_name(self) -> str:
        """
        Get a human-readable name for the solver combination.

        Returns
        -------
        str
            A string describing the job and AGV solver combination.
            Format: "Job:<job_solver_name>+AGV:<agv_solver_name>"

        Example:
            >>> solver = UniversalSolver(instance, SPT(), SCPT(), num_agvs=3)
            >>> solver.get_name()
            'Job:SPT+AGV:SCPT'
        """
        job_name = self._get_solver_name(self.job_solver, self.job_observation_kwargs)
        agv_name = self._get_solver_name(self.agv_solver, self.agv_observation_kwargs)
        if agv_name == job_name:
            return f"M:{agv_name}"
        return f"Job:{job_name}+AGV:{agv_name}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the solver."""
        job_name = self._get_solver_name(self.job_solver)
        agv_name = self._get_solver_name(self.agv_solver)
        return (
            f"UniversalSolver(job_solver={job_name}, agv_solver={agv_name}, "
            f"num_agvs={self.num_agvs}, device={self.device})"
        )

    def __str__(self) -> str:
        """Return a concise string representation of the solver."""
        return self.get_name()

    def get_model_path(self) -> str | None:
        """
        Get the model path if the solver is ML-based.

        Returns
        -------
        str | None
            The model path if either job_solver or agv_solver is ML-based,
            otherwise None.
        """
        j_p = None
        a_p = None
        if self.job_is_ml and hasattr(self.job_observation_kwargs, "model_path"):
            j_p = self.job_observation_kwargs.model_path
        else:
            j_p = self.job_solver
        if self.agv_is_ml and hasattr(self.agv_observation_kwargs, "model_path"):
            a_p = self.agv_observation_kwargs.model_path
        else:
            a_p = self.agv_solver
        return j_p, a_p

    def solve(
        self, instance: JSSPTransportInstance, num_agvs: int
    ) -> tuple[int, TransportSchedule]:
        """
        Solve the transport instance using the configured solvers.

        The solve function iterates until the schedule is complete:
        1. Get the next job to schedule (from heuristic or ML model)
        2. Get the AGV to assign (from heuristic or ML model)
        3. Schedule the job with the selected AGV
        4. Repeat until schedule is complete

        Returns
        -------
        tuple[int, TransportSchedule]
            A tuple containing:
            - makespan: The makespan of the completed schedule
            - schedule: The completed TransportSchedule object

        Example:
            solver = UniversalSolver(instance, job_solver, agv_solver, num_agvs=3)
            makespan, schedule = solver.solve()
            print(f"Final makespan: {makespan}")
        """
        self.instance = instance
        self.num_agvs = num_agvs
        self.schedule = TransportSchedule(instance, num_agvs=num_agvs)
        # Reset schedule to initial state
        self.reset()

        # Main scheduling loop
        while not self.schedule.is_complete():
            # Step 1: Get job action
            if self.job_is_heuristic:
                # Heuristic: call step method directly
                job_id = self.job_solver.step(self.schedule)
            elif self.job_is_ml:
                # ML: get observation and run through policy
                current_job_observation = self.job_observation_provider.get_observation(
                    self.schedule
                )
                current_job_observation = (
                    current_job_observation["node_feats"],
                    current_job_observation["edge_index"],
                )

                action_mask = self.schedule.get_valid_operation_mask()
                # Get logits from policy
                with torch.no_grad():
                    self.job_solver.eval()
                    logits = self.job_solver(
                        current_job_observation[0],
                        current_job_observation[1],
                        action_mask,
                    )

                    # Apply action mask
                    masked_logits = logits.masked_fill(
                        ~torch.tensor(
                            action_mask, dtype=torch.bool, device=self.device
                        ),
                        float("-inf"),
                    )

                    # Validate shape
                    expected_shape = (1, self.schedule.num_operations)
                    assert (
                        masked_logits.shape == expected_shape
                        or masked_logits.shape == logits.shape
                    ), (
                        f"masked_logits shape mismatch: expected {expected_shape}, "
                        f"got {masked_logits.shape}"
                    )
                    probabilities = torch.softmax(masked_logits, dim=-1)
                    selected_job_id = int(torch.argmax(probabilities).item())

                    # Convert to job_id if necessary

                    job_id = self.schedule.flat_index_to_job_op(selected_job_id)[0]

            else:
                raise ValueError(
                    f"job_solver must be either a Heuristic or torch.nn.Module, got {type(self.job_solver)}"
                )
            # Store selected job_id for AGV selector
            self.action_selected_job_id = job_id
            self.set_selected_job_aec(job_id)
            # Step 2: Get AGV action
            if self.agv_is_heuristic:
                # Heuristic: call step method with job_id
                agv_id = self.agv_solver.step(self.schedule, job_id)
            elif self.agv_is_ml:
                # ML: get observation and run through policy
                agv_observation = self.agv_observation_provider.get_observation(
                    self.schedule
                )
                agv_obs_tensor = torch.as_tensor(
                    agv_observation["feat"], dtype=torch.float32, device=self.device
                )

                # Get logits from policy
                with torch.no_grad():
                    self.agv_solver.eval()
                    agv_logits = self.agv_solver(agv_obs_tensor).view(1, -1)

                    # Get AGV action mask (all AGVs are typically valid)
                    action_mask = np.ones(self.num_agvs, dtype=np.int8)

                    # Apply action mask
                    masked_logits = agv_logits.masked_fill(
                        ~torch.tensor(
                            action_mask, dtype=torch.bool, device=self.device
                        ),
                        float("-inf"),
                    )

                    # Validate shape
                    expected_shape = (1, self.schedule.num_agvs)
                    assert masked_logits.shape == expected_shape, (
                        f"masked_logits shape mismatch: expected {expected_shape}, "
                        f"got {masked_logits.shape}"
                    )

                    probabilities = torch.softmax(masked_logits, dim=-1)
                    agv_id = int(torch.argmax(probabilities).item())
            else:
                raise ValueError(
                    f"agv_solver must be either a Heuristic or torch.nn.Module, got {type(self.agv_solver)}"
                )

            # Step 3: Schedule the job with the selected AGV
            success = self.schedule.schedule_job(job_id, agv=agv_id)

            if not success:
                raise RuntimeError(
                    f"Failed to schedule job {job_id} with AGV {agv_id}. "
                    f"This may indicate an invalid action from the solver."
                )

        # Return makespan and final schedule
        makespan = self.schedule.get_makespan()
        self.action_selected_job_id = None
        return makespan, self.schedule

    def get_schedule(self) -> TransportSchedule:
        """
        Get the current schedule.

        Returns
        -------
        TransportSchedule
            The current schedule state
        """
        return self.schedule

    def reset(self) -> None:
        """
        Reset the solver to solve a new instance.

        This recreates the schedule from the original instance and
        resets observation providers if using ML models.
        """
        self.schedule = TransportSchedule(self.instance, num_agvs=self.num_agvs)

        if self.job_observation_provider is not None:
            self.job_observation_provider.reset(self.schedule)
        if self.agv_observation_provider is not None:
            self.agv_observation_provider.reset(self.schedule)

    def set_selected_job_aec(self, action: int) -> None:
        """
        Set the selected job for AEC (Asynchronous Environment Communication).

        This is used to communicate the selected job to the schedule,
        which may be needed by the AGV selector's observation provider.

        Parameters
        ----------
        action : int
            The selected job ID
        """
        self.schedule.set_selected_job_aec(action)


def _get_job_scheduler_from_gnn_dispatcher(
    gnn_solver_path_policy: str,
) -> torch.nn.Module:
    gnn_solver_params = torch.load(f"{gnn_solver_path_policy}")
    gnn_solver_path = gnn_solver_path_policy.split("checkpoints")[0]
    gnn_cfg = load_policy_config(Path(f"{gnn_solver_path}/.hydra/config.yaml"))
    job_selector_attr = hasattr(gnn_cfg.finetune, "job_selector")
    if not job_selector_attr:
        apply_lora = gnn_cfg.finetune.lora
        gnn_cfg_finetune_kwargs = gnn_cfg.finetune
    else:
        if gnn_cfg.finetune.job_selector:
            apply_lora = gnn_cfg.finetune.job_selector_kwargs.lora
        else:
            apply_lora = False

    if apply_lora:
        gnn_solver_params = lora_params(gnn_solver_params)

    if not job_selector_attr:
        gnn_cfg_finetune_kwargs.finetune = False
        gnn_cfg_finetune_kwargs.lora = False
    else:
        gnn_cfg.finetune.job_selector = False
    gnn_dispatcher = GNNDispatcher(cfg=gnn_cfg)

    gnn_dispatcher.setup_instance()
    gnn_dispatcher.setup_environment()

    policy_tuple, value_module = gnn_dispatcher.setup_models()

    policy_tuple[1].load_state_dict(gnn_solver_params, strict=True)
    job_solver = policy_tuple[0]["job_selector"].module[0]
    # Load parameters into the job_solver
    job_observatioin_provider = copy.deepcopy(gnn_dispatcher.env.observation_provider)
    job_observatioin_provider_kwargs = copy.deepcopy(
        gnn_dispatcher.env.job_observation_kwargs
    )
    job_observatioin_provider_kwargs["model_path"] = gnn_solver_path_policy
    return job_solver, job_observatioin_provider, job_observatioin_provider_kwargs


def _get_agv_scheduler_from_gnn_dispatcher(
    gnn_solver_path_policy: str,
) -> torch.nn.Module:
    gnn_solver_params = torch.load(f"{gnn_solver_path_policy}")
    gnn_solver_path = gnn_solver_path_policy.split("checkpoints")[0]
    gnn_cfg = load_policy_config(Path(f"{gnn_solver_path}/.hydra/config.yaml"))

    job_selector_attr = hasattr(gnn_cfg.finetune, "job_selector")
    if not job_selector_attr:
        apply_lora = gnn_cfg.finetune.lora
        gnn_cfg_finetune_kwargs = gnn_cfg.finetune
    else:
        if gnn_cfg.finetune.job_selector:
            apply_lora = gnn_cfg.finetune.job_selector_kwargs.lora
        else:
            apply_lora = False

    if apply_lora:
        gnn_solver_params = lora_params(gnn_solver_params)

    if not job_selector_attr:
        gnn_cfg_finetune_kwargs.finetune = False
        gnn_cfg_finetune_kwargs.lora = False
    else:
        gnn_cfg.finetune.job_selector = False
    gnn_dispatcher = GNNDispatcher(cfg=gnn_cfg)
    gnn_dispatcher.setup_instance()
    gnn_dispatcher.setup_environment()
    policy_tuple, value_module = gnn_dispatcher.setup_models()
    policy_tuple[1].load_state_dict(gnn_solver_params)
    agv_solver = policy_tuple[0]["agv_selector"].module[0]
    # Load parameters into the agv_solver
    agv_observatioin_provider = copy.deepcopy(
        gnn_dispatcher.env.agv_observation_provider
    )
    agv_observatioin_provider_kwargs = copy.deepcopy(
        gnn_dispatcher.env.agv_observation_kwargs
    )
    agv_observatioin_provider_kwargs["model_path"] = gnn_solver_path_policy
    return agv_solver, agv_observatioin_provider, agv_observatioin_provider_kwargs


def _get_agv_scheduler_from_agv_dispatcher(
    agv_solver_path_policy: str,
) -> torch.nn.Module:
    agv_solver_params = torch.load(f"{agv_solver_path_policy}")
    agv_solver_path = agv_solver_path_policy.split("checkpoints")[0]
    agv_cfg = load_policy_config(Path(f"{agv_solver_path}/.hydra/config.yaml"))
    agv_dispatcher = AgvDispatcher(
        cfg=agv_cfg, job_selector=job_heuristic_factory("spt")
    )
    agv_dispatcher.setup_instance()
    agv_dispatcher.setup_environment()
    policy_tuple, value_module = agv_dispatcher.setup_models()
    policy_tuple[1].load_state_dict(agv_solver_params)
    agv_solver = policy_tuple[0]["agv_selector"].module[0]
    # Load parameters into the agv_solver

    agv_observatioin_provider = copy.deepcopy(
        agv_dispatcher.env.agv_observation_provider
    )
    agv_observatioin_provider_kwargs = copy.deepcopy(
        agv_dispatcher.env.agv_observation_kwargs
    )
    agv_observatioin_provider_kwargs["model_path"] = agv_solver_path_policy
    return agv_solver, agv_observatioin_provider, agv_observatioin_provider_kwargs


def _get_job_scheduler_from_agv_dispatcher(
    agv_solver_path_policy: str,
) -> torch.nn.Module:
    agv_solver_path = agv_solver_path_policy.split("checkpoints")[0]
    agv_cfg = load_policy_config(Path(f"{agv_solver_path}/.hydra/config.yaml"))
    if agv_cfg.agv.ml_policy:
        attr_heuristic = hasattr(agv_cfg.agv, "heuristic")
        if not attr_heuristic:
            # Old apporach with load from FM
            if "ml_path" in agv_cfg.agv:
                ml_path = agv_cfg.agv.ml_path
                fm_model_path = ml_path.split("checkpoints")[0]
            else:
                fm_date = agv_cfg.agv.date
                name = agv_cfg.agv.name
                fm_model_path = "outputs/" + fm_date + "/" + name
                ml_path = fm_model_path + "/checkpoints/best_model.pt"
                print(ml_path)
                print(
                    "Warning: Using FM model path from AGV config. Consider updating to use ml_path."
                )
            job_solver, job_observatioin_provider, job_observatioin_provider_kwargs = (
                _get_job_scheduler_from_fm_solver(ml_path)
            )

        else:
            # New approach
            job_ml_path = agv_cfg.agv.ml_path
            job_solver, job_observatioin_provider, job_observatioin_provider_kwargs = (
                _get_job_scheduler_from_jssp_solver(job_ml_path)
            )
    elif not agv_cfg.agv.ml_policy and agv_cfg.agv.heuristic:
        print("Using heuristic for JOB scheduler", agv_cfg.agv.heuristic_name)
        heuristic = job_heuristic_factory(agv_cfg.agv.heuristic_name)
        return heuristic, None, None

    # fm_model_path = agv_solver_path.replace(name, fm_date)

    return job_solver, job_observatioin_provider, job_observatioin_provider_kwargs


def _get_job_scheduler_from_jssp_solver(
    gnn_solver_path_policy: str,
) -> torch.nn.Module:
    job_solver_params = torch.load(f"{gnn_solver_path_policy}")
    job_solver_path = gnn_solver_path_policy.split("checkpoints")[0]
    job_cfg = load_policy_config(Path(f"{job_solver_path}/.hydra/config.yaml"))
    job_dispatcher = JsspDispatcher(
        cfg=job_cfg, agv_selector=agv_heuristic_factory("scpt")
    )
    job_dispatcher.setup_instance()
    job_dispatcher.setup_environment()
    policy_tuple, value_module = job_dispatcher.setup_models()
    policy_tuple[1].load_state_dict(job_solver_params, strict=True)
    job_solver = policy_tuple[0]["job_selector"].module[0]
    job_observatioin_provider = copy.deepcopy(job_dispatcher.env.observation_provider)
    job_observatioin_provider_kwargs = copy.deepcopy(
        job_dispatcher.env.job_observation_kwargs
    )
    job_observatioin_provider_kwargs["model_path"] = gnn_solver_path_policy
    return job_solver, job_observatioin_provider, job_observatioin_provider_kwargs


def _get_agv_scheduler_from_jssp_solver(
    agv_solver_path_policy: str,
) -> torch.nn.Module:
    agv_solver_path = agv_solver_path_policy.split("checkpoints")[0]
    agv_cfg = load_policy_config(Path(f"{agv_solver_path}/.hydra/config.yaml"))
    if not agv_cfg.jssp.ml_policy and agv_cfg.jssp.heuristic:
        print("Using heuristic for AGV scheduler", agv_cfg.jssp.heuristic_name)
        heuristic = agv_heuristic_factory(agv_cfg.jssp.heuristic_name)
        return heuristic, None, None
    elif agv_cfg.jssp.ml_policy:
        agv_model_path = agv_cfg.jssp.ml_path
        agv_solver, policy_cfg = _get_agv_selector(agv_model_path)

        agv_dispatcher = JsspDispatcher(cfg=agv_cfg, agv_selector=agv_solver)
        agv_dispatcher.setup_instance()
        agv_dispatcher.setup_environment()
        agv_observatioin_provider = copy.deepcopy(
            agv_dispatcher.env.agv_observation_provider
        )
        agv_observatioin_provider_kwargs = copy.deepcopy(
            policy_cfg.env.agv_observation_kwargs
        )
        agv_observatioin_provider_kwargs["model_path"] = agv_model_path
        return agv_solver, agv_observatioin_provider, agv_observatioin_provider_kwargs
