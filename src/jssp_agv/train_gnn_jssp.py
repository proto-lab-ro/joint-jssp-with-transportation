import copy

import hydra
from omegaconf import DictConfig

from jssp_gnn.utils import get_device, set_device


set_device("cpu")
from jssp_agv.dispatcher import JsspDispatcher
from jssp_agv.evaluator import create_agv_evaluator
from jssp_agv.mlp_train.train_mappo import create_test_env, train_mappo
from jssp_agv.train_gnn_agv import (
    _get_policy_and_config_path,
)
from jssp_agv.utils.load_model import (
    load_state_dict_and_extract_agv_policy,
    recreate_agv_model_from_config,
)
from jssp_core import set_seed
from jssp_core.solver.heuristics import agv_heuristic_factory
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
        "agv_observation_provider" not in pol_cfg_env
        and "agv_observation_provider" in pol_cfg_env
    ):
        cfg.env.agv_observation_provider = pol_cfg_env.agv_observation_provider
    if (
        "agv_observation_kwargs" not in pol_cfg_env
        and "agv_observation_kwargs" in pol_cfg_env
    ):
        cfg.env.agv_observation_kwargs = pol_cfg_env.agv_observation_kwargs
    return cfg


def _get_agv_selector(ml_path):
    policy_path, config_path, policy_cfg = _get_policy_and_config_path(ml_path)

    combined_policy, env_type = recreate_agv_model_from_config(policy_cfg)
    agv_selector = load_state_dict_and_extract_agv_policy(
        combined_policy, policy_path, device, None
    )

    return agv_selector, policy_cfg


@hydra.main(version_base=None, config_path="../../conf/agv", config_name="default_jssp")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra for configuration management."""
    print("=============================================")
    print("Starting JSSP Training With Fixed AGV Selector")
    print("=============================================")

    log_dir, save_dir = setup_directories(cfg)

    logger = TrainingLogger(log_dir)
    checkpoint_logger = ModelCheckpointLogger(save_dir, logger)

    if cfg.jssp.ml_policy:
        ml_path = cfg.jssp.ml_path
        agv_selector, policy_cfg = _get_agv_selector(ml_path)

        cfg = _adapt_config_based_on_loaded_policy(cfg, policy_cfg)

    elif not cfg.jssp.ml_policy and cfg.jssp.heuristic:
        agv_selector = agv_heuristic_factory(cfg.jssp.heuristic_name)

    # ======
    log_training_config(cfg)

    TrainDispatcher = JsspDispatcher(cfg, agv_selector)

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
