import torch
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import set_exploration_type


class AECCollector(SyncDataCollector):
    """A data collector for AEC (Agent-Environment-Cycle) environments.

    This collector overrides the rollout method to properly handle AEC environments
    where agents act sequentially rather than in parallel. In AEC environments,
    each agent takes turns acting, and the environment progresses one agent at a time.

    The key difference from the standard SyncDataCollector is that this collector
    respects the agent iteration cycle of AEC environments.

    Args:
        create_env_fn: Callable that returns an AEC environment instance.
        policy: Policy to be executed in the environment.
        frames_per_batch (int): Number of frames to collect per batch.
        total_frames (int): Total number of frames to collect.
        **kwargs: Additional keyword arguments passed to SyncDataCollector.
    """

    @torch.no_grad()
    def rollout(self):
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        if self.reset_at_each_iter:
            self._shuttle.update(self.env.reset())

        if self._use_buffers:
            self._final_rollout.fill_(("collector", "traj_ids"), -1)
        else:
            pass
        tensordicts = []

        agv_selector_ob_key = ("agv_selector", "observation")
        job_selector_ob_key = ("job_selector", "observation")

        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                policy_input = self._shuttle

                # =================== AEC specific logic ==========================

                all_agents = self.env.agents
                for agent in range(len(all_agents)):
                    acting_agent = self.env.agent_selection
                    group_key = next(
                        (
                            k
                            for k, agents in self.env.group_map.items()
                            if acting_agent in agents
                        ),
                        None,
                    )
                    if group_key is None:
                        raise KeyError(f"No group key contains agent {acting_agent}")

                    acting_agent = group_key
                    policy_output = self.policy[acting_agent](policy_input)

                    if self._shuttle is not policy_output:
                        # ad-hoc update shuttle
                        self._shuttle.update(
                            policy_output, keys_to_update=self._policy_output_keys
                        )

                    env_input = self._shuttle
                    env_output, env_next_output = self.env.step_and_maybe_reset(
                        env_input
                    )
                    if acting_agent == "job_selector":
                        self._shuttle[agv_selector_ob_key] = env_next_output[
                            agv_selector_ob_key
                        ]

                    policy_input = env_next_output

                if self._shuttle is not env_output:
                    next_data = env_output.get("next")
                    if self._shuttle_has_no_device:
                        next_data.clear_device_()
                    self._shuttle.set("next", next_data)

                if self.storing_device is not None:
                    non_blocking = (
                        not self.no_cuda_sync or self.storing_device.type == "cuda"
                    )
                    tensordicts.append(
                        self._shuttle.to(self.storing_device, non_blocking=non_blocking)
                    )
                    if not self.no_cuda_sync:
                        self._sync_storage()
                else:
                    tensordicts.append(self._shuttle)

                collector_data = self._shuttle.get("collector").copy()
                self._shuttle = env_next_output
                if self._shuttle_has_no_device:
                    self._shuttle.clear_device_()
                self._shuttle.set("collector", collector_data)
                # =================== End AEC specific logic ==========================
                self._update_traj_ids(env_output)

            else:
                if self._use_buffers:
                    result = self._final_rollout
                    try:
                        result = torch.stack(
                            tensordicts,
                            self._final_rollout.ndim - 1,
                            out=self._final_rollout,
                        )

                    except RuntimeError:
                        with self._final_rollout.unlock_():
                            result = torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout,
                            )
                elif self.replay_buffer is not None and not self.extend_buffer:
                    return
                else:
                    result = TensorDict.maybe_dense_stack(tensordicts, dim=-1)
                    result.refine_names(..., "time")

        return self._maybe_set_truncated(result)
