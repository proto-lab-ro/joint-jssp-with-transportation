import copy
from pathlib import Path

from jssp_gnn.utils import get_device, set_device


set_device("cpu")
import hydra
from omegaconf import DictConfig

from jssp_agv.dispatcher import AgvDispatcher
from jssp_agv.evaluator import create_agv_evaluator
from jssp_agv.mlp_train.train_mappo import create_test_env, train_mappo
from jssp_agv.utils.load_model import (
    load_policy_config,
    load_state_dict_and_extract_job_policy,
    recreate_job_model_from_config,
)
from jssp_core import set_seed
from jssp_core.solver.heuristics import job_heuristic_factory
from jssp_core.utils.utils import log_training_config, setup_directories
from jssp_gnn.logger import ModelCheckpointLogger, TrainingLogger


set_seed(42)
device = get_device()


def _adapt_config_based_on_loaded_policy(
    cfg: DictConfig, policy_cfg: DictConfig
) -> DictConfig:
    """Adapt training configuration based on loaded policy configuration for compatibility."""
    pol_cfg_env = policy_cfg.env
    if (
        "job_observation_provider" not in pol_cfg_env
        and "observation_provider" in pol_cfg_env
    ):
        cfg.env.job_observation_provider = pol_cfg_env.observation_provider
    if (
        "job_observation_kwargs" not in pol_cfg_env
        and "observation_kwargs" in pol_cfg_env
    ):
        cfg.env.job_observation_kwargs = pol_cfg_env.observation_kwargs

    cfg.env.job_selector_type = policy_cfg.env.job_selector_type

    return cfg


def _get_policy_and_config_path(ml_path):
    policy_path = Path(ml_path)
    config_path = Path(ml_path.split("checkpoints")[0]) / ".hydra" / "config.yaml"
    print(f"Loading policy from: {policy_path}")
    print(f"Loading config from: {config_path}")
    policy_cfg = load_policy_config(config_path)

    return policy_path, config_path, policy_cfg


def _get_job_selector(ml_path):
    policy_path, config_path, policy_cfg = _get_policy_and_config_path(ml_path)

    combined_policy, env_type = recreate_job_model_from_config(policy_cfg)
    job_selector = load_state_dict_and_extract_job_policy(
        combined_policy, policy_path, device, env_type
    )

    return job_selector, policy_cfg


@hydra.main(version_base=None, config_path="../../conf/agv", config_name="default_agv")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra for configuration management."""
    print("=============================================")
    print("Starting AGV Training With Fixed Job Selector")
    print("=============================================")

    log_dir, save_dir = setup_directories(cfg)
    logger = TrainingLogger(log_dir)
    checkpoint_logger = ModelCheckpointLogger(save_dir, logger)

    if cfg.agv.ml_policy:
        ml_path = cfg.agv.ml_path
        job_selector, policy_cfg = _get_job_selector(ml_path)
        cfg = _adapt_config_based_on_loaded_policy(cfg, policy_cfg)
    # Load Heuristic
    elif not cfg.agv.ml_policy and cfg.agv.heuristic:
        job_selector = job_heuristic_factory(cfg.agv.heuristic_name)

    # ======
    log_training_config(cfg)

    TrainDispatcher = AgvDispatcher(cfg, job_selector)

    testEnv = create_test_env(TrainDispatcher)

    Evaluator = create_agv_evaluator(
        env=testEnv,
        logger=logger,
        cfg=copy.deepcopy(cfg),
        dispatcher=copy.deepcopy(TrainDispatcher),
    )
    if cfg.training.inside:
        set_seed(42)
        trained_policy = train_mappo(
            cfg=cfg,
            logger=logger,
            checkpoint_logger=checkpoint_logger,
            dispatcher=TrainDispatcher,
            evaluator=Evaluator,
        )
    else:
        raise NotImplementedError("Outside training is not implemented yet.")

    return trained_policy


if __name__ == "__main__":
    main()
