import torch
from omegaconf import DictConfig
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import ActionMask

from jssp_agv.modules import (
    SharedGraphFeatureExtractor,
)
from jssp_agv.modules.graph_matrix import (
    SharedGraphFeatureExtractorMatrix,
)
from jssp_core.domain import MarlEnvType, ObservationType
from jssp_core.environments import AgvEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Shared Utilities
# ============================================================================
from jssp_agv.dispatcher.utils_mlp import _basic_env_kwargs


def get_env_kwargs(cfg: DictConfig, job_selector):
    """Extract environment kwargs from configuration."""

    job_selector_kwargs = {
        "job_observation_provider": cfg.env.job_observation_provider,
        "job_observation_kwargs": cfg.env.job_observation_kwargs,
        "job_selector_type": cfg.env.job_selector_type,
    }
    basic_env_kwargs = _basic_env_kwargs(cfg)
    basic_env_kwargs["job_selector"] = job_selector
    basic_env_kwargs["job_selector_kwargs"] = job_selector_kwargs

    return basic_env_kwargs


# ============================================================================
# MLP-Specific Functions
# ============================================================================


def create_environment(
    cfg: DictConfig, instance=None, job_selector=None
) -> TransformedEnv:
    """Create standard MLP-based environment."""
    env_kwargs = get_env_kwargs(cfg, job_selector)
    use_mask = False

    if "env_type" in cfg.marl:
        marl_env_type = cfg.marl.env_type
        # print("Creating environment of type:", marl_env_type)

        if marl_env_type == MarlEnvType.PARALLEL:
            raw_env = AgvEnv(instance=instance, **env_kwargs)
            use_mask = False
            env = PettingZooWrapper(
                env=raw_env, return_state=False, group_map=None, use_mask=use_mask
            )

        elif marl_env_type == MarlEnvType.AEC:
            raise ValueError(
                'MarlEnvType.AEC not SUpported for this Environment: Expected "parallel".'
            )
        else:
            raise ValueError(
                f'Unknown env_type: "{marl_env_type}". Expected "parallel" or "aec".'
            )

        env = TransformedEnv(
            env,
            Compose(
                ActionMask(
                    mask_key=("agv_selector", "action_mask"),
                    action_key=("agv_selector", "action"),
                ),
                StepCounter(max_steps=cfg.training.max_steps),
                RewardSum(),
            ),
        )
    else:
        raise ValueError('Missing env_type. Expected "parallel or aec".')

    return env


def create_shared_extractor(env, cfg: DictConfig):
    """Create GNN-based policies with shared feature extractor."""
    policies = {}
    shared_extractor = None
    node_feats_dim = None
    obs_p = env.job_observation_provider
    node_feats_dim = obs_p.get_observation_space()["node_feats"].shape[1]
    if env.job_policy_observation_type == ObservationType.GRAPH:
        shared_extractor = SharedGraphFeatureExtractor(
            input_dim=node_feats_dim,
            hidden_dim=cfg.gnn_feature_extractor.get("hidden_dim", 64),
            k_layers=cfg.gnn_feature_extractor.get("k_layers", 3),
        )

    elif env.job_policy_observation_type == ObservationType.GRAPH_MATRIX:
        shared_extractor = SharedGraphFeatureExtractorMatrix(
            input_dim=node_feats_dim,
            hidden_dim=cfg.gnn_feature_extractor.get("hidden_dim", 64),
            k_layers=cfg.gnn_feature_extractor.get("k_layers", 3),
        )

    return shared_extractor
