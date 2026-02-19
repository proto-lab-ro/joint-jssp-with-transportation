"""
LoRA Fine-tuning for GNN-based Multi-Agent JSSP-AGV.

This script loads a pre-trained GNN model and applies LoRA (Low-Rank Adaptation)
for parameter-efficient fine-tuning on new JSSP instances.
"""

import copy
from pathlib import Path

import hydra
from omegaconf import DictConfig

from jssp_gnn.utils import get_device, set_device


set_device("cpu")
from torch import nn

from jssp_agv.dispatcher import GNNDispatcher
from jssp_agv.evaluator import create_agv_evaluator
from jssp_agv.mlp_train.train_mappo import create_test_env, train_mappo
from jssp_agv.train_gnn_agv import _adapt_config_based_on_loaded_policy as _acbold_job
from jssp_agv.train_gnn_jssp import _adapt_config_based_on_loaded_policy as _acbold_agv
from jssp_agv.utils.load_model import (
    load_policy_config,
    load_state_dict_and_extract_agv_policy,
    load_state_dict_and_extract_job_policy,
    recreate_job_model_from_config,
)
from jssp_core import set_seed
from jssp_core.utils.utils import log_training_config, setup_directories
from jssp_gnn.logger import ModelCheckpointLogger, TrainingLogger


set_seed(42)
device = get_device()


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


def recreate_agv_model_from_config(cfg: DictConfig) -> nn.Module:
    """Recreate the model from the configuration file."""

    dispatcher = GNNDispatcher(cfg)

    instance = dispatcher.setup_instance()
    env = dispatcher.setup_environment()
    (policies, combined_policies), _ = dispatcher.setup_models()
    return combined_policies, None


def _get_agv_selector(ml_path):
    policy_path, config_path, policy_cfg = _get_policy_and_config_path(ml_path)

    combined_policy, env_type = recreate_agv_model_from_config(policy_cfg)
    agv_selector = load_state_dict_and_extract_agv_policy(
        combined_policy, policy_path, device, None
    )

    return agv_selector, policy_cfg


@hydra.main(
    version_base=None, config_path="../../conf/agv", config_name="default_aec_gnn"
)
def main(cfg: DictConfig) -> None:
    """Main LoRA fine-tuning function."""

    log_dir, save_dir = setup_directories(cfg)

    # Finetuning Start=========================
    if cfg.finetune.job_selector:
        ml_path_jssp = cfg.finetune.job_selector_kwargs.ml_path
        job_selector, job_policy_cfg = _get_job_selector(ml_path_jssp)
        cfg = _acbold_job(cfg, job_policy_cfg)

    if cfg.finetune.agv_selector:
        ml_path_agv = cfg.finetune.agv_selector_kwargs.ml_path
        agv_selector, agv_policy_cfg = _get_agv_selector(ml_path_agv)
        cfg = _acbold_agv(cfg, agv_policy_cfg)

    log_training_config(cfg)
    logger = TrainingLogger(log_dir)
    checkpoint_logger = ModelCheckpointLogger(save_dir, logger)

    TrainDispatcher = GNNDispatcher(cfg)

    # Set Models for Finetuning =========================
    if cfg.finetune.job_selector:
        TrainDispatcher.set_loaded_pre_trained_job_policy(job_selector)
    if cfg.finetune.agv_selector:
        TrainDispatcher.set_loaded_pre_trained_agv_policy(agv_selector)

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
