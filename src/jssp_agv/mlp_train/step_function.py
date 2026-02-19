import torch


def batch_optimization_step(
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
):
    if idx == 0:
        frames = agent_reset_frames
    batch = batch.to(device)
    loss_vals = group_loss_module(batch)
    loss_value = (
        loss_vals["loss_objective"]
        + loss_vals["loss_critic"]
        + loss_vals["loss_entropy"]
    )

    # Backward pass and optimization
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(
        group_loss_module.parameters(),
        cfg.training[agent_group].max_grad_norm
        if agent_group in cfg.training
        else cfg.training.max_grad_norm,
    )

    group_optimizer.step()
    group_optimizer.zero_grad()

    frames += batch.numel()
    if False:
        logger.log_loss_metrics(
            loss_value=loss_value.item(),
            kl_approx=loss_vals["kl_approx"].item(),
            clip_fraction=loss_vals["clip_fraction"].item(),
            loss_objective=loss_vals["loss_objective"].item(),
            loss_critic=loss_vals["loss_critic"].item(),
            step=frames,
            agent=agent_group,
        )
    num_updates += 1

    return frames, num_updates


def log_train_step(tensordict_data, env, logger, optimizer, rollout_frames):
    mask = tensordict_data["next", "done"]
    terminated = tensordict_data.get(("next", "terminated"))

    for group in env.group_map.keys():
        mean_reward = tensordict_data["next", group, "reward"].mean().item()
        sum_reward = tensordict_data["next", group, "reward"].sum().item()
        # Log training metrics
        logger.log_training_lr(
            learning_rate=optimizer[group].param_groups[0]["lr"],
            step=rollout_frames,
        )
        if False:
            logger.log_training_metrics(
                learning_rate=optimizer[group].param_groups[0]["lr"],
                mean_reward=mean_reward,
                sum_reward=sum_reward,
                step=rollout_frames,
            )

        logger.log_episode_metrics(
            max_return=tensordict_data["next", group, "episode_reward"].max().item(),
            avg_return=tensordict_data["next", group, "episode_reward"][mask]
            .mean()
            .item(),
            max_length=tensordict_data["step_count"].max().item(),
            min_makespan=tensordict_data["next", group, "info", "makespan"][terminated]
            .min()
            .item(),
            mean_makespan=tensordict_data["next", group, "info", "makespan"][terminated]
            .mean()
            .item(),
            count_terminated=(terminated.sum().item() if terminated is not None else 0),
            num_agvs=env.num_agvs,
            step=rollout_frames,
        )
        break
