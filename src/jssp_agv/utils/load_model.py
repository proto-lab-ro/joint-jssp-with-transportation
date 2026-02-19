from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from jssp_agv.dispatcher import (
    AgvDispatcher,
    GNNDispatcher,
    JsspDispatcher,
)
from jssp_core.domain import EnvironmentType
from jssp_core.domain.domains import ObservationType
from jssp_core.solver.heuristics import agv_heuristic_factory, job_heuristic_factory


def recreate_agv_model_from_config(cfg: DictConfig) -> nn.Module:
    """Recreate the model from the configuration file."""

    dispatcher = AgvDispatcher(cfg, job_selector=job_heuristic_factory("mor"))

    instance = dispatcher.setup_instance()
    env = dispatcher.setup_environment()
    (policies, combined_policies), _ = dispatcher.setup_models()
    return combined_policies, None


def load_state_dict_and_extract_agv_policy(
    combined_policies, policy_path, device, env_type=None
) -> nn.Module:
    """Load the policy weights and extract the job policy module.

    Args:
        combined_policies: The policy module (ProbabilisticActor for SINGLE_AGENT, dict for MULTI_AGENT)
        policy_path: Path to the saved policy weights
        device: Device to load the model on
        env_type: Type of environment (SINGLE_AGENT or MULTI_AGENT)

    Returns:
        The extracted job policy module in eval mode
    """
    state_dict = torch.load(policy_path, map_location=device)
    combined_policies.load_state_dict(state_dict)

    agv_selector = combined_policies["agv_selector"]
    agv_policy = agv_selector[0].module.eval()

    return agv_policy


def recreate_job_model_from_config(cfg: DictConfig) -> nn.Module:
    """Recreate the model from the configuration file."""

    env = cfg.env
    if "job_observation_kwargs" not in env and "observation_kwargs" in env:
        job_observation_kwargs = env.observation_kwargs
        env_type = EnvironmentType.SINGLE_AGENT
    else:
        job_observation_kwargs = env.job_observation_kwargs
        env_type = EnvironmentType.MULTI_AGENT

    hast_attr = hasattr(cfg, "marl")
    if hast_attr:
        if cfg.marl.env_type == "parallel":
            env_type = EnvironmentType.MULTI_AGENT

            dispatcher = JsspDispatcher(cfg, agv_selector=agv_heuristic_factory("scpt"))
            instance = dispatcher.setup_instance()
            env = dispatcher.setup_environment()
            (policies, combined_policies), _ = dispatcher.setup_models()
            return combined_policies, env_type

    if not hast_attr or cfg.marl.env_type != "parallel":
        if job_observation_kwargs["observation_type"] == ObservationType.FLAT:
            raise NotImplementedError("FLAT observation type not implemented yet.")

        elif job_observation_kwargs["observation_type"] == ObservationType.DICT:
            raise NotImplementedError("DICT observation type not implemented yet.")

        elif job_observation_kwargs["observation_type"] == ObservationType.GRAPH:
            if env_type == EnvironmentType.SINGLE_AGENT:
                raise NotImplementedError(
                    "GRAPH observation type with SINGLE_AGENT not implemented yet."
                )
            elif env_type == EnvironmentType.MULTI_AGENT:
                dispatcher = GNNDispatcher(cfg)

        elif job_observation_kwargs["observation_type"] == ObservationType.GRAPH_MATRIX:
            if env_type == EnvironmentType.SINGLE_AGENT:
                raise NotImplementedError(
                    "GRAPH_MATRIX observation type with SINGLE_AGENT not implemented yet."
                )
            elif env_type == EnvironmentType.MULTI_AGENT:
                dispatcher = GNNDispatcher(cfg)

        instance = dispatcher.setup_instance()
        env = dispatcher.setup_environment()

        if env_type == EnvironmentType.SINGLE_AGENT:
            policy_module, _ = dispatcher.setup_models()

            return policy_module, env_type

        elif env_type == EnvironmentType.MULTI_AGENT:
            (policies, combined_policies), _ = dispatcher.setup_models()
            return combined_policies, env_type


def load_state_dict_and_extract_job_policy(
    combined_policies, policy_path, device, env_type
) -> nn.Module:
    """Load the policy weights and extract the job policy module.

    Args:
        combined_policies: The policy module (ProbabilisticActor for SINGLE_AGENT, dict for MULTI_AGENT)
        policy_path: Path to the saved policy weights
        device: Device to load the model on
        env_type: Type of environment (SINGLE_AGENT or MULTI_AGENT)

    Returns:
        The extracted job policy module in eval mode
    """
    combined_policies.load_state_dict(torch.load(policy_path, map_location=device))

    if env_type == EnvironmentType.SINGLE_AGENT:
        job_policy = combined_policies.module[0].module.eval()

    elif env_type == EnvironmentType.MULTI_AGENT:
        job_selector = combined_policies["job_selector"]
        job_policy = job_selector.module[0].module.eval()

    return job_policy


def load_policy_config(config_path: Path) -> DictConfig:
    """Load the configuration file for the policy."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(str(config_path))
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    return cfg
