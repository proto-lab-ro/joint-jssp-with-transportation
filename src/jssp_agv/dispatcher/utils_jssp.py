from omegaconf import DictConfig
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import ActionMask

from jssp_agv.environments.agv import GnnJssp_AGVEnv
from jssp_core.domain import MarlEnvType
from jssp_gnn.utils import get_device


device = get_device()
# ============================================================================
# Shared Utilities
# ============================================================================


def _basic_env_kwargs(cfg: DictConfig):
    """Extract basic environment kwargs from configuration."""
    env_kwargs = {
        "max_episode_steps": cfg.env.max_episode_steps,
        "random_instance": cfg.env.random_instance,
        "reward_function": cfg.env.reward_function,
        "reward_kwargs": cfg.env.reward_kwargs,
        "job_observation_provider": cfg.env.job_observation_provider,
        "job_observation_kwargs": cfg.env.job_observation_kwargs,
        "job_selector_type": cfg.env.job_selector_type,
        "number_agvs": cfg.env.number_agvs,
        "instance_generator": cfg.env.instance_generator,
        "instance_generator_kwargs": cfg.env.instance_generator_kwargs,
    }
    return env_kwargs


def get_env_kwargs(cfg: DictConfig, agv_selector):
    """Extract environment kwargs from configuration."""

    agv_selector_kwargs = {
        "agv_observation_provider": cfg.env.agv_observation_provider,
        "agv_observation_kwargs": cfg.env.agv_observation_kwargs,
    }
    basic_env_kwargs = _basic_env_kwargs(cfg)
    basic_env_kwargs["agv_selector"] = agv_selector
    basic_env_kwargs["agv_selector_kwargs"] = agv_selector_kwargs
    return basic_env_kwargs


# ============================================================================
# MLP-Specific Functions
# ============================================================================


def create_environment(
    cfg: DictConfig, instance=None, agv_selector=None
) -> TransformedEnv:
    """Create standard MLP-based environment."""
    env_kwargs = get_env_kwargs(cfg, agv_selector)
    use_mask = False

    if "env_type" in cfg.marl:
        marl_env_type = cfg.marl.env_type

        if marl_env_type == MarlEnvType.PARALLEL:
            raw_env = GnnJssp_AGVEnv(instance=instance, **env_kwargs)
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
                    mask_key=("job_selector", "action_mask"),
                    action_key=("job_selector", "action"),
                ),
                StepCounter(max_steps=cfg.training.max_steps),
                RewardSum(),
            ),
        )
    else:
        raise ValueError('Missing env_type. Expected "parallel or aec".')

    return env
