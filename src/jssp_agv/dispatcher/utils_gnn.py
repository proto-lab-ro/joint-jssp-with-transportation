import torch
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.modules import MaskedCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from jssp_agv.dispatcher.utils import get_critic_keys_based_on_env, get_env_transforms
from jssp_agv.environments.agv import GnnJsspAGVEnvAEC

# from jssp_agv.gnn_pettingzoo_wrapper import AgvGnnPettingZooWrapper
from jssp_agv.modules import (
    AgvGnnPettingZooWrapper,
    MlpAgvActor,
    SB3LikeCriticMatrix_ohneAGV,
    SB3LikeJobActor,
    SharedGraphFeatureExtractor,
    get_matrix_critic,
)
from jssp_agv.modules.graph_matrix import (
    SB3LikeActorMatrix,
    SharedGraphFeatureExtractorMatrix,
)
from jssp_core.domain import MarlEnvType, ObservationType
from jssp_core.environments import JSSP_AGV
from jssp_gnn.utils import get_device


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
# ============================================================================
# Shared Utilities
# ============================================================================


from jssp_agv.dispatcher.utils_mlp import (
    get_env_kwargs,
)


# ============================================================================
# GNN-Specific Functions
# ============================================================================


def _get_gnn_environment(cfg: DictConfig, instance=None, env_kwargs=None):
    marl_env_type = cfg.marl.env_type

    if marl_env_type == MarlEnvType.PARALLEL:
        raise NotImplementedError("Parallel GNN environment not implemented yet.")

    elif marl_env_type == MarlEnvType.AEC:
        raw_env = GnnJsspAGVEnvAEC(instance=instance, **env_kwargs)
        use_mask = True
        env = AgvGnnPettingZooWrapper(
            env=raw_env,
            return_state=False,
            group_map=raw_env.group_map,
            use_mask=use_mask,
        )
    else:
        raise ValueError(
            f'Unknown env_type: "{marl_env_type}". Expected "parallel" or "aec".'
        )
    return env


def create_gnn_environment(cfg: DictConfig, instance=None):
    """Create GNN-based environment."""
    env_kwargs = get_env_kwargs(cfg)
    use_mask = False

    if "env_type" in cfg.marl:
        env = _get_gnn_environment(cfg, instance, env_kwargs)
        standard_transform = get_env_transforms(cfg)
        standard_transform.append(
            RewardSum([("agv_selector", "reward"), ("job_selector", "reward")])
        )

        env = TransformedEnv(
            env,
            standard_transform,
        )
    else:
        raise ValueError('Missing env_type. Expected "parallel or aec".')

    return env


def create_agv_mlp_policy(env, cfg: DictConfig = None):
    policies = {}
    for agent_group, agents in env.group_map.items():
        agv_actor = _get_agv_mlp_policy(env, agent_group)
        policies[agent_group] = agv_actor
    combined_policies = TensorDictSequential(policies)
    if cfg is not None:
        if cfg.training.get("continue", False):
            combined_policies.load_state_dict(
                torch.load(cfg.training.continiue_model_path, map_location=device)
            )
    return policies, combined_policies


def _get_agv_mlp_policy(env, agent_group: str):
    obs_spec = env.observation_spec[(agent_group, "observation")]
    try:
        # Composite-like spec: iterate over children
        observation_dim_agent = 0
        for _, child in obs_spec.items():
            if hasattr(child, "shape") and child.shape and child.shape[-1] is not None:
                observation_dim_agent += child.shape[-1]
    except Exception:
        # Fallback for non-composite spec
        observation_dim_agent = obs_spec.shape[-1]

    in_keys_agv_agent = env.get_agent_observation_in_keys(agent_group)

    in_key = [(agent_group, "observation", key) for key in in_keys_agv_agent]
    policy = TensorDictModule(
        MlpAgvActor(
            input_dim=observation_dim_agent,
        ),
        in_keys=in_key,
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
    return policy_actor


def create_gnn_policies(env, cfg: DictConfig):
    """Create GNN-based policies with shared feature extractor."""
    policies = {}
    shared_extractor = None

    for agent_group, agents in env.group_map.items():
        if agent_group.startswith("agv"):
            policy_actor = _get_agv_mlp_policy(env, agent_group)
            policies[agent_group] = policy_actor

        elif agent_group.startswith("job"):
            # Create shared GNN feature extractor
            # num_jobs = env.num_jobs
            # num_operations = env.num_operations

            if env.get_observation_type("job_selector_0") == ObservationType.GRAPH:
                shared_extractor = SharedGraphFeatureExtractor(
                    input_dim=env.observation_space(agents[0])["node_feats"].shape[1],
                    hidden_dim=cfg.gnn_feature_extractor.get("hidden_dim", 64),
                    k_layers=cfg.gnn_feature_extractor.get("k_layers", 3),
                )
                sb_actor = SB3LikeJobActor(
                    shared_extractor=shared_extractor,
                    forward_type=env.job_selector_type,
                )
                actor_in_keys = [(agent_group, "observation")]

            elif (
                env.get_observation_type("job_selector_0")
                == ObservationType.GRAPH_MATRIX
            ):
                shared_extractor = SharedGraphFeatureExtractorMatrix(
                    input_dim=env.observation_space(agents[0])["node_feats"].shape[1],
                    hidden_dim=cfg.gnn_feature_extractor.get("hidden_dim", 64),
                    k_layers=cfg.gnn_feature_extractor.get("k_layers", 3),
                )
                sb_actor = SB3LikeActorMatrix(
                    shared_extractor=shared_extractor,
                    forward_type=env.job_selector_type,
                )

                actor_in_keys = [
                    (agent_group, "observation", "node_feats"),
                    (agent_group, "observation", "edge_index"),
                    (agent_group, "action_mask"),
                ]

            # Create actor with shared extractor
            policy_module = TensorDictModule(
                sb_actor,
                in_keys=actor_in_keys,
                out_keys=[(agent_group, "logits")],
            )

            policy_actor = ProbabilisticActor(
                module=policy_module,
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

        else:
            raise ValueError(
                f'Unknown agent group: "{agent_group}". '
                'Expected to start with "agv" or "job".'
            )

    combined_policies = TensorDictSequential(policies)

    if cfg.training.get("continue", False):
        combined_policies.load_state_dict(
            torch.load(cfg.training.continiue_model_path, map_location=device)
        )
    return policies, combined_policies, shared_extractor


def create_gnn_critic(env, shared_extractor, cfg: DictConfig):
    """Create GNN-based critic with shared feature extractor."""

    if isinstance(env._env, (JSSP_AGV)):
        agv_in_dim = 0
    else:
        agv_in_dim = env.observation_spec[("agv_selector", "observation")].shape[-1]
    value_mode_in_keys, obs_type = get_critic_keys_based_on_env(env)

    if obs_type == ObservationType.GRAPH:
        raise NotImplementedError("Graph critic not implemented yet.")
        sb_critic = SB3LikeCriticMatrix_ohneAGV(
            shared_extractor=shared_extractor,
            agv_in_dim=agv_in_dim,
        )

    elif obs_type == ObservationType.GRAPH_MATRIX:
        critic_kswargs = {
            "shared_extractor": shared_extractor,
            "agv_in_dim": agv_in_dim,
            "obs_type": obs_type,
        }
        sb_critic = get_matrix_critic(cfg.env.critic, **critic_kswargs)

    value_module = TensorDictModule(
        sb_critic,
        in_keys=value_mode_in_keys,
        out_keys=["state_value"],
    )

    value_module = ValueOperator(
        module=value_module,
        in_keys=value_mode_in_keys,
        out_keys=["state_value"],
    )

    return value_module


def create_gnn_loss_modules(policies, value_module, cfg: DictConfig):
    """Create PPO loss modules with integrated GAE for GNN."""
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

        deactivate_vmap = True
        loss_module.make_value_estimator(
            ValueEstimators.GAE,
            gamma=cfg.training.gamma,
            lmbda=cfg.training.lmbda,
            average_gae=cfg.training.average_gae,
            deactivate_vmap=deactivate_vmap,
        )

        loss_modules[agent] = loss_module

    return loss_modules
