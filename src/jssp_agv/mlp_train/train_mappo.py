import copy
import multiprocessing as mp

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from jssp_agv.curriculum import get_agv_curriculum_manager
from jssp_agv.dispatcher.utils_mlp import process_batch
from jssp_agv.mlp_train.step_function import batch_optimization_step, log_train_step
from jssp_agv.modules.lora import count_lora_parameters, merge_all_lora_weights
from jssp_core import set_seed
from jssp_core.early_stopping import create_early_stopping
from jssp_gnn.evaluator import BaseEvaluator
from jssp_gnn.logger import ModelCheckpointLogger, TrainingLogger
from jssp_gnn.utils import get_device


device = get_device()
set_seed(42)


def train_mappo(
    cfg: DictConfig,
    logger: TrainingLogger,
    checkpoint_logger: ModelCheckpointLogger,
    dispatcher,
    evaluator: BaseEvaluator | None = None,
):
    # Load Instance
    _ = dispatcher.setup_instance()

    # Create environment and models
    env = dispatcher.setup_environment()
    policy_module, value_module = dispatcher.setup_models()

    # Create other Components
    replay_buffer = dispatcher.setup_replay_buffer()
    loss_module, advantage_module = dispatcher.setup_loss_modules()
    optimizer = dispatcher.setup_optimizers()
    collector = dispatcher.setup_collector()
    scheduler = dispatcher.setup_schedulers()

    pbar = tqdm(total=cfg.training.total_frames)
    num_updates = 0
    frames = 0
    rollout_frames = 0

    curriculum_manager = get_agv_curriculum_manager(
        cfg,
        dispatcher=dispatcher,
    )
    es_cfg = cfg.training.get("early_stopping", {})
    stopper = create_early_stopping(es_cfg)
    print("Active children:", mp.active_children())
    total_rollouts = 0
    if evaluator is not None:
        _ = evaluator.evaluate(policy_module[1], rollout_frames)
    try:
        while rollout_frames < cfg.training.total_frames:
            if curriculum_manager.should_update(rollout_frames):
                env, collector, replay_buffer = curriculum_manager.update(
                    rollout_frames, env, collector, replay_buffer, policy_module
                )

            for i, tensordict_data in enumerate(collector):
                total_rollouts += 1
                epoch_frames = frames
                number_frames = tensordict_data.numel()
                rollout_frames += number_frames
                epoch_frames_step_size = number_frames // cfg.training.num_epochs

                # ===============================================
                # Trennung bei MARL
                # ===============================================

                for epoch_idx in range(cfg.training.num_epochs):
                    epoch_reset_frames = frames
                    for agent_group, agents_in_group in env.group_map.items():
                        group_batch = tensordict_data.to(device)
                        group_batch = process_batch(group_batch, agent_group)
                        group_batch = group_batch.reshape(-1)
                        group_buffer = replay_buffer[agent_group]

                        group_loss_module = loss_module[agent_group]
                        group_optimizer = optimizer[agent_group]
                        group_scheduler = scheduler[agent_group]
                        with torch.no_grad():
                            group_loss_module.value_estimator(
                                group_batch,
                                params=group_loss_module.critic_network_params,
                                target_params=group_loss_module.target_critic_network_params,
                            )

                        group_buffer.extend(group_batch.to(device))
                        agent_reset_frames = epoch_reset_frames

                        for idx, batch in enumerate(group_buffer):
                            frames, num_updates = batch_optimization_step(
                                idx,
                                batch,
                                agent_reset_frames,
                                frames,
                                cfg,
                                group_loss_module,
                                group_optimizer,
                                agent_group,
                                device,
                                logger,
                                num_updates,
                            )
                        group_scheduler.step()
                    epoch_frames += epoch_frames_step_size

                # ===============================================
                # Ende Trennung bei MARL
                # ===============================================
                # Episode and training metrics logging

                log_train_step(
                    tensordict_data,
                    env,
                    logger,
                    optimizer,
                    rollout_frames,
                )

                if evaluator is not None:
                    current_eval_freq = cfg.training.get("eval_freq", 0)

                    if current_eval_freq is not None and number_frames > 0:
                        if current_eval_freq % number_frames != 0:
                            current_eval_freq = (
                                round(current_eval_freq / number_frames) * number_frames
                            )
                            if current_eval_freq == 0:
                                current_eval_freq = number_frames
                    should_eval = False
                    if current_eval_freq is not None and evaluator.should_evaluate(
                        rollout_frames, current_eval_freq
                    ):
                        should_eval = True

                    if (
                        cfg.training.get("eval_freq", 0) is not None
                        and total_rollouts % cfg.training.get("eval_rollout_freq", 0)
                        == 0
                    ):
                        should_eval = True
                    if should_eval:
                        eval_metrics = evaluator.evaluate(
                            policy_module[1], rollout_frames
                        )
                        es_cfg = cfg.training.get("early_stopping", {})

                        if es_cfg and eval_metrics:
                            metric_name = es_cfg.metric_name
                            current_value = eval_metrics.get(metric_name)
                            if current_value is not None:
                                print(
                                    f"Early stopping check for metric '{metric_name}': {current_value}"
                                )
                                if stopper.check_stop(current_value):
                                    print(
                                        f"Early stopping triggered at {rollout_frames} frames "
                                        f"for metric '{metric_name}' with value {current_value}."
                                    )
                                    rollout_frames = cfg.training.total_frames
                                    break
                pbar.update(tensordict_data.numel())

                if curriculum_manager.should_update(rollout_frames):
                    break
    finally:
        try:
            if collector is not None:
                collector.shutdown()

        except Exception as e:
            print(f"[cleanup] collector shutdown error: {e}", flush=True)

        try:
            if env is not None and hasattr(env, "close"):
                env.close()
        except Exception as e:
            print(f"[cleanup] env.close() error: {e}", flush=True)

        try:
            if evaluator is not None:
                evaluator.close()
        except Exception as e:
            print(f"[cleanup] evaluator.close() error: {e}", flush=True)
    pbar.close()

    lora_params = count_lora_parameters(policy_module)
    if lora_params > 0:
        policy_module_copy = copy.deepcopy(policy_module)
        merge_all_lora_weights(policy_module_copy)

        checkpoint_logger.save_final_checkpoint(
            policy_module_copy, "policy_module_final.pt"
        )
    else:
        checkpoint_logger.save_final_checkpoint(policy_module, "policy_module_final.pt")

    logger.close()

    return policy_module


def create_test_env(dispatcher):
    _ = dispatcher.setup_instance()
    test_env = dispatcher.setup_test_environment()
    return test_env
