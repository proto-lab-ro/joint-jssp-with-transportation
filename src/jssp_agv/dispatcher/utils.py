from jssp_agv.environments.agv import GnnJsspAGVEnvAEC
from jssp_core.domain import ObservationType
from jssp_core.environments import JSSP_AGV, AgvEnv


BOTH_GNN_MATRIX_KEYS = [
    ("job_selector", "observation"),
    ("agv_selector", "observation"),
]
BOTH_GNN_MATRIX_CRITIC_KEYS = [
    ("job_selector", "observation", "node_feats"),
    ("job_selector", "observation", "edge_index"),
    ("agv_selector", "observation"),
]
ONLY_AGV_GNN_MATRIX_CRITIC_KEYS = [
    ("agv_selector", "info", "critic_jssp_node_feats"),
    ("agv_selector", "info", "critic_jssp_edge_index"),
    ("agv_selector", "observation"),
]

ONLY_JSSP_GNN_MATRIX_CRITIC_KEYS = [
    ("job_selector", "observation", "node_feats"),
    ("job_selector", "observation", "edge_index"),
    ("job_selector", "observation", "node_feats"),
]


def get_critic_keys_based_on_env(env):
    """Get critic input keys based on environment type and observation type."""
    if isinstance(env._env, AgvEnv):
        if env.job_policy_observation_type == ObservationType.GRAPH:
            raise NotImplementedError("Graph critic not implemented yet.")
        elif env.job_policy_observation_type == ObservationType.GRAPH_MATRIX:
            return ONLY_AGV_GNN_MATRIX_CRITIC_KEYS, ObservationType.GRAPH_MATRIX

    elif isinstance(env._env, (GnnJsspAGVEnvAEC)):
        if env.get_observation_type("job_selector_0") == ObservationType.GRAPH:
            return BOTH_GNN_MATRIX_KEYS, ObservationType.GRAPH
        elif env.get_observation_type("job_selector_0") == ObservationType.GRAPH_MATRIX:
            return BOTH_GNN_MATRIX_CRITIC_KEYS, ObservationType.GRAPH_MATRIX

    elif isinstance(env._env, (JSSP_AGV)):
        if env.get_observation_type("job_selector_0") == ObservationType.GRAPH:
            raise NotImplementedError("Graph critic not implemented yet.")
        elif env.get_observation_type("job_selector_0") == ObservationType.GRAPH_MATRIX:
            return ONLY_JSSP_GNN_MATRIX_CRITIC_KEYS, ObservationType.GRAPH_MATRIX

    return None


import copy

from omegaconf import DictConfig
from torchrl.envs import (
    Compose,
    StepCounter,
)
from torchrl.envs.transforms import ActionMask


STANDARD_ENV_TRANSFORMS = Compose(
    ActionMask(
        mask_key=("job_selector", "action_mask"),
        action_key=("job_selector", "action"),
    ),
    ActionMask(
        mask_key=("agv_selector", "action_mask"),
        action_key=("agv_selector", "action"),
    ),
    # RewardSum(),
)


def get_env_transforms(cfg: DictConfig):
    """Get standard environment transforms."""
    standard_t = copy.deepcopy(STANDARD_ENV_TRANSFORMS)
    standard_t.append(StepCounter(max_steps=cfg.training.max_steps))
    return standard_t


import re


def clean_key(key: str):
    # Remove any ".original_XXX" path segment completely
    return re.sub(r"\.original_[^.]+", "", key)


def lora_params(param_dict):
    new_sd = {}
    for key, param in param_dict.items():
        if "lora" in key:
            continue
        elif "original" in key:
            new_sd[clean_key(key)] = param
        else:
            new_sd[key] = param
    param_dict = new_sd

    clean_sd = {}

    for k, v in param_dict.items():
        # Remove the wrong policy_head namespace
        if (
            "module.job_selector.module.0.policy_head" in k
            and "module.job_selector.module.0.module.policy_head" not in k
        ):
            continue

        clean_sd[k] = v

    param_dict = clean_sd

    return param_dict
