import math
from pathlib import Path

import torch
from omegaconf import DictConfig
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MLP, MaskedCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from jssp_agv.environments.agv import GnnJsspAGVEnvAEC
from jssp_agv.modules.aec_collector import AECCollector
from jssp_core.environments import JSSP_AGV, AgvEnv, JsspAGVEnvAEC
from jssp_core.instances import JSSPInstance, JSSPTransportInstance
from jssp_gnn.utils import get_device


device = get_device()
# ============================================================================
# Shared Utilities
# ============================================================================


def get_instance(cfg: DictConfig):
    """Load problem instance from configuration."""
    instance_path = cfg.env.instance
    p = Path(instance_path)
    if "transport" in p.parts:
        instance = JSSPTransportInstance.from_file(str(p) + ".yaml")
    else:
        instance = JSSPInstance.from_file(str(p))

    return instance


def _basic_env_kwargs(cfg: DictConfig):
    """Extract basic environment kwargs from configuration."""
    env_kwargs = {
        "max_episode_steps": cfg.env.max_episode_steps,
        "random_instance": cfg.env.random_instance,
        "reward_function": cfg.env.reward_function,
        "reward_kwargs": cfg.env.reward_kwargs,
        "agv_observation_provider": cfg.env.agv_observation_provider,
        "agv_observation_kwargs": cfg.env.agv_observation_kwargs,
        "number_agvs": cfg.env.number_agvs,
        "instance_generator": cfg.env.instance_generator,
        "instance_generator_kwargs": cfg.env.instance_generator_kwargs,
    }
    return env_kwargs


def get_env_kwargs(cfg: DictConfig):
    """Extract environment kwargs from configuration."""
    basic_env_kwargs = _basic_env_kwargs(cfg)

    additional_kwargs = {
        "job_observation_provider": cfg.env.job_observation_provider,
        "job_observation_kwargs": cfg.env.job_observation_kwargs,
        "job_selector_type": cfg.env.job_selector_type,
    }
    return basic_env_kwargs | additional_kwargs


# ============================================================================
# MLP-Specific Functions
# ============================================================================


def create_policy(env, cfg: DictConfig):
    """Create MLP-based policies for each agent group."""
    policies = {}
    for agent_group, agents in env.group_map.items():
        observation_dim_agent = env.observation_spec[
            (agent_group, "observation")
        ].shape[-1]
        action_dim_agent = env.action_spec[(agent_group, "action")].space.n

        policy = TensorDictModule(
            MLP(observation_dim_agent, action_dim_agent, depth=2),
            in_keys=[(agent_group, "observation")],
            out_keys=[(agent_group, "logits")],
        )

        policy_actor = ProbabilisticActor(
            module=policy,
            spec=env.action_spec[agent_group, "action"],
            in_keys={
                "logits": (agent_group, "logits"),
                "mask": (agent_group, "action_mask"),
            },
            out_keys=[(agent_group, "action")],
            distribution_class=MaskedCategorical,
            return_log_prob=True,
            log_prob_key=(agent_group, "log_prob"),
        )

        policies[agent_group] = policy_actor

    combined_policies = TensorDictSequential(policies)
    return policies, combined_policies


def create_critics(env, cfg: DictConfig):
    """Create MLP-based critic."""
    sum_dim_shapes = 0
    in_keys_agent_groups = []

    for agent_group, agents in env.group_map.items():
        sum_dim_shapes += env.observation_spec[(agent_group, "observation")].shape[-1]
        in_keys_agent_groups.append((agent_group, "observation"))

    value_module = TensorDictModule(
        MLP(sum_dim_shapes, 1, depth=2),
        in_keys=in_keys_agent_groups,
        out_keys=["state_value"],
    )

    value_module = ValueOperator(
        module=value_module,
        in_keys=in_keys_agent_groups,
        out_keys=["state_value"],
    )

    return value_module


def create_loss_modules(policies, value_module, cfg: DictConfig):
    """Create PPO loss modules with integrated GAE."""
    loss_modules = {}

    for agent, policy in policies.items():
        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=value_module,
            clip_epsilon=cfg.training[agent].clip_epsilon
            if agent in cfg.training
            else cfg.training.clip_epsilon,
            entropy_bonus=bool(cfg.training.entropy_eps),
            entropy_coeff=cfg.training.entropy_eps,
            critic_coeff=cfg.training.critic_coef,
            loss_critic_type=cfg.training.loss_critic_type,
            normalize_advantage=cfg.training.normalize_advantage,
        )
        loss_module.set_keys(
            reward=(agent, "reward"),
            action=(agent, "action"),
            done=(agent, "done"),
            terminated=(agent, "terminated"),
            advantage=(agent, "advantage"),
            value_target=(agent, "value_target"),
            value=("state_value"),
            sample_log_prob=(agent, "log_prob"),
        )

        loss_module.make_value_estimator(
            ValueEstimators.GAE,
            gamma=cfg.training.gamma,
            lmbda=cfg.training.lmbda,
            average_gae=cfg.training.average_gae,
        )

        loss_modules[agent] = loss_module

    return loss_modules


# ============================================================================
# Shared Training Components
# ============================================================================


def create_collectors(env, seq_policy, cfg: DictConfig):
    """Create data collector for experience gathering."""

    if isinstance(env.base_env._env, (AgvEnv, JSSP_AGV)):
        collector = SyncDataCollector(
            env,
            seq_policy,
            frames_per_batch=cfg.training.frames_per_batch,
            total_frames=cfg.training.total_frames,
            split_trajs=cfg.training.split_trajs,
            device=device,
            reset_at_each_iter=True,
        )
        return collector

    elif isinstance(env.base_env._env, (JsspAGVEnvAEC, GnnJsspAGVEnvAEC)):
        collector = AECCollector(
            env,
            seq_policy,
            frames_per_batch=cfg.training.frames_per_batch,
            total_frames=cfg.training.total_frames,
            split_trajs=cfg.training.split_trajs,
            device=device,
            reset_at_each_iter=True,
        )
        return collector


def create_schedulers(optimizers, cfg: DictConfig):
    """Create learning rate schedulers."""
    schedulers = {}

    if cfg.curriculum.interval.type == "number":
        total_rollouts = math.ceil(
            cfg.training.total_frames / cfg.training.lr_frames_per_batch
        )
    else:
        total_rollouts = math.ceil(
            cfg.training.total_frames / cfg.training.frames_per_batch
        )
    # once per epoch
    decay_steps = cfg.training.get(
        "lr_decay_steps",
        cfg.training.num_epochs * total_rollouts,
    )

    for agent, optimizer in optimizers.items():
        schedulers[agent] = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(0.0, 1.0 - step / max(1, decay_steps)),
        )
    return schedulers


def create_optimizers(loss_modules, cfg: DictConfig):
    """Create optimizers for each agent."""
    optimizers = {}
    for agent, loss_module in loss_modules.items():
        optimizers[agent] = torch.optim.Adam(
            loss_module.parameters(),
            lr=cfg.training[agent].lr if agent in cfg.training else cfg.training.lr,
        )
    return optimizers


def create_replay_buffer(env, cfg: DictConfig):
    """Create replay buffers for each agent group."""
    replay_buffers = {}
    for group, _agents in env.group_map.items():
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=cfg.training.frames_per_batch),
            sampler=SamplerWithoutReplacement(shuffle=True),
            batch_size=cfg.training.sub_batch_size,
        )
        replay_buffers[group] = replay_buffer

    return replay_buffers


def process_batch(batch: TensorDictBase, group: str):
    keys = list(batch.keys(True, True))
    group_shape = batch.get(group).shape
    nested_done_key = ("next", group, "done")
    nested_terminated_key = ("next", group, "terminated")
    nested_reward_key = ("next", group, "reward")

    if nested_done_key not in keys:
        batch.set(
            nested_done_key,
            batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
        )
    if nested_terminated_key not in keys:
        batch.set(
            nested_terminated_key,
            batch.get(("next", "terminated")).unsqueeze(-1).expand((*group_shape, 1)),
        )

    if nested_reward_key not in keys:
        batch.set(
            nested_reward_key,
            batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
        )

    return batch
