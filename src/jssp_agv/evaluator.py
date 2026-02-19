import copy

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from jssp_agv.modules.lora import count_lora_parameters, merge_all_lora_weights
from jssp_core.domain.base_dispatcher import DispatcherBase
from jssp_gnn.evaluator import (
    BaseEvaluator,
    StandardEvaluator,
    create_evaluator,
    create_test_env,
)
from jssp_gnn.logger import TrainingLogger


class MultiAgentBenchmark(BaseEvaluator):
    ACTION_RELATED_KEYS = ["action", "log_prob", "logits", "action_mask"]

    def __init__(
        self,
        env: TransformedEnv,
        logger: TrainingLogger,
        max_steps: int = 1000,
        save_best_model: bool = False,
        save_dir: str | None = None,
        metric_mode: str = "max",
        metric_key: str = "avg_return",
        cfg: DictConfig | None = None,
        dispatcher=None,
        detailed_eval: bool = True,
    ):
        """
        Initialize benchmark evaluator.

        Args:
            env: The environment to evaluate on
            logger: Logger for recording evaluation metrics
            instance_generator: Generator that yields EvaluationInstance objects
            n_instances: Number of instances to evaluate
            max_steps: Maximum steps for evaluation rollout
            save_best_model: Whether to store the best-performing model checkpoint
            save_dir: Directory where the checkpoint is written
            metric_mode: "min" or "max" comparison strategy for metric_key
            metric_key: Aggregated metric from evaluate() to track
        """
        super().__init__(env, logger)
        self.dispatcher = dispatcher
        self.max_steps = max_steps
        self.save_best_model = save_best_model
        self.save_dir = save_dir
        self.metric_mode = metric_mode
        self.metric_key = metric_key
        self.cfg = cfg
        self.is_aec_env = self._check_aec_environment()
        self.detailed_eval = detailed_eval

        if metric_mode not in {"min", "max"}:
            raise ValueError(f"metric_mode must be 'min' or 'max', got {metric_mode}")

        self.best_metric = float("inf") if metric_mode == "min" else -float("inf")
        self.best_step: int | None = None

        if save_best_model and save_dir is None:
            raise ValueError("save_dir must be provided when save_best_model=True")

    def _check_aec_environment(self) -> bool:
        """
        Check if the environment is an AEC (Agent Environment Cycle) type.

        Returns:
            True if the environment is AEC type, False otherwise
        """
        if self.cfg is not None and hasattr(self.cfg, "marl"):
            env_type = self.cfg.marl.get("env_type", None)
            return env_type == "aec"
        return False

    def transform_td(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Transform tensordict for AEC environments.

        This function filters out duplicate observations and restructures the tensordict
        to match the expected format for multi-agent evaluation. It assumes that:
        - Odd indices (1, 3, 5, ...) contain AGV-job decisions
        - Even indices (0, 2, 4, ...) contain job-job decisions
        - We keep only the odd indices (AGV-job decisions)

        Args:
            tensordict: The raw tensordict from rollout

        Returns:
            Transformed tensordict with filtered data
        """
        agv_selector_next_obs_key = ("next", "agv_selector", "observation")

        all_indices_of_agv_job = np.arange(
            tensordict[agv_selector_next_obs_key].shape[0]
        )[1::2]
        all_indices_of_job_job = np.arange(
            tensordict[agv_selector_next_obs_key].shape[0]
        )[0::2]

        for action_key in self.ACTION_RELATED_KEYS:
            all_action_of_job_job = tensordict[("job_selector", action_key)][
                all_indices_of_job_job
            ]
            tensordict[("job_selector", action_key)][all_indices_of_agv_job] = (
                all_action_of_job_job
            )

        new_tensorDict = tensordict[1::2]
        return new_tensorDict

    def evaluate(self, policy_module, step: int) -> dict[str, float]:
        """
        Run evaluation with multi-agent group structure.

        Args:
            policy_module: The policy to evaluate
            step: Current training step

        Returns:
            Dictionary of evaluation metrics aggregated across all agent groups
        """

        eval_instances = [
            "jssp_instances/transport/ft06",  # 6x6
            "jssp_instances/transport/ft10",  # 10x10
            "jssp_instances/transport/ft20",  # 20x5
            "jssp_instances/transport/la24",  # 15x10
            "jssp_instances/transport/la23",  # 15x10
            "jssp_instances/transport/la15",  # 20x5
            "jssp_instances/transport/la32",  # 30x10
        ]
        results = dict()
        for instance_path in eval_instances:
            for num_agvs in _list_agvs(instance_path):
                stage = {
                    "number_agvs": num_agvs,
                    "instance": instance_path,
                }
                self.cfg.env = OmegaConf.merge(self.cfg.env, OmegaConf.create(stage))
                self.dispatcher.cfg = self.cfg
                self.env = create_test_env(self.dispatcher)

                all_metrics = self.eval_one_instance(policy_module, step)
                results[(instance_path, num_agvs)] = all_metrics

        first_group = list(self.env.group_map.keys())[0]

        aggregated_metrics = dict()
        mean_makespan = 0.0
        avg_return = 0.0
        for (instance_path, num_agvs), all_metrics in results.items():
            mean_makespan += all_metrics[first_group]["makespan"]
            avg_return += all_metrics[first_group]["sum_reward"]
        aggregated_metrics["mean_makespan"] = mean_makespan / len(results)
        aggregated_metrics["avg_return"] = avg_return / len(results)

        self.logger.log_benchmark_evaluation_metrics(
            avg_return=aggregated_metrics["avg_return"],
            makespan=aggregated_metrics["mean_makespan"],
            step=step,
        )

        if self.detailed_eval:
            instance_metrics: dict[str, dict[str, float]] = {}
            for (instance_path, num_agvs), all_metrics in results.items():
                if instance_path not in instance_metrics:
                    instance_metrics[instance_path] = {
                        "sum_rewards": [],
                        "makespans": [],
                    }
                instance_metrics[instance_path]["sum_rewards"].append(
                    all_metrics[first_group]["sum_reward"]
                )
                instance_metrics[instance_path]["makespans"].append(
                    all_metrics[first_group]["makespan"]
                )

            detailed_results: dict[tuple[str, int], dict[str, float]] = {}
            for instance_path, metrics_data in instance_metrics.items():
                mean_sum_reward = sum(metrics_data["sum_rewards"]) / len(
                    metrics_data["sum_rewards"]
                )
                mean_makespan = sum(metrics_data["makespans"]) / len(
                    metrics_data["makespans"]
                )
                detailed_results[(instance_path, 0)] = {
                    "mean_sum_reward": mean_sum_reward,
                }

            self.logger.log_detailed_benchmark_metrics(
                step=step,
                all_results=detailed_results,
            )

        if self.save_best_model:
            current_metric = aggregated_metrics.get(self.metric_key)
            if current_metric is not None:
                is_best = False
                if self.metric_mode == "min":
                    is_best = current_metric < self.best_metric
                else:
                    is_best = current_metric > self.best_metric

                if is_best:
                    self.best_metric = current_metric
                    self.best_step = step

                    lora_params = count_lora_parameters(policy_module)
                    if lora_params > 0:
                        policy_module_copy = copy.deepcopy(policy_module)
                        merge_all_lora_weights(policy_module_copy)

                        self._save_model(policy_module_copy, step, current_metric)
                    else:
                        policy_module_copy = copy.deepcopy(policy_module)
                        self._save_model(policy_module_copy, step, current_metric)

                    self.logger.log_custom_metrics(
                        {f"eval/best_{self.metric_key}": self.best_metric},
                        step=step,
                    )
                    print(
                        f"New best {self.metric_key}: {self.best_metric:.4f} at step {step}"
                    )

        return aggregated_metrics

    def _save_model(self, model, step: int, metric_value: float):
        """Save the model checkpoint for benchmark tracking."""
        import os

        assert self.save_dir is not None
        os.makedirs(self.save_dir, exist_ok=True)

        if isinstance(model, tuple):
            separate_models = model[0]
            main_model = model[1]

            for key, small_model in separate_models.items():
                checkpoint_path = os.path.join(self.save_dir, f"{key}_best_model.pt")
                torch.save(small_model.state_dict(), checkpoint_path)

            main_checkpoint_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(main_model.state_dict(), main_checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)

        metadata_path = os.path.join(self.save_dir, "best_model_metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"Best {self.metric_key}: {metric_value:.6f}\n")
            f.write(f"Step: {step}\n")
            f.write(f"Mode: {self.metric_mode}\n")

    def eval_one_instance(self, policy_module, step):
        with (
            set_exploration_type(ExplorationType.DETERMINISTIC),
            torch.no_grad(),
        ):
            eval_rollout = self.env.rollout(
                self.max_steps,
                policy_module,
            )

            if self.is_aec_env:
                eval_rollout = self.transform_td(eval_rollout)

            eval_mask = eval_rollout["next", "done"]
            eval_terminated = eval_rollout["next", "terminated"]

            all_metrics = {}

            for group in self.env.group_map.keys():
                group_metrics = {
                    "mean_reward": eval_rollout["next", group, "reward"].mean().item(),
                    "sum_reward": eval_rollout["next", group, "reward"].sum().item(),
                    "step_count": eval_rollout["step_count"].max().item(),
                    "avg_return": eval_rollout["next", group, "episode_reward"][
                        eval_mask
                    ]
                    .mean()
                    .item(),
                    "makespan": eval_rollout["next", group, "info", "makespan"][
                        eval_terminated
                    ].item(),
                }

                all_metrics[group] = group_metrics
                break

            del eval_rollout
        return all_metrics


class MultiAgentEvaluator(StandardEvaluator):
    """
    Multi-agent evaluator that handles group-based tensordict structure.

    This evaluator is designed for multi-agent environments where rewards and metrics
    are organized by agent groups (e.g., "jobs", "agvs"). It extracts metrics from
    nested tensordict entries like eval_rollout["next", group, "reward"].

    For AEC (Agent Environment Cycle) environments, it automatically applies the
    transform_td function to filter and restructure the evaluation rollout data.
    """

    ACTION_RELATED_KEYS = ["action", "log_prob", "logits", "action_mask"]

    def __init__(
        self,
        env: TransformedEnv,
        logger: TrainingLogger,
        max_steps: int = 1000,
        lower_bound: float | None = None,
        save_best_model: bool = True,
        save_dir: str | None = None,
        metric_mode: str = "min",
        metric_key: str = "makespan",
        cfg: DictConfig | None = None,
    ):
        """
        Initialize multi-agent evaluator.

        Args:
            env: The environment to evaluate on
            logger: Logger for recording evaluation metrics
            max_steps: Maximum steps for evaluation rollout
            lower_bound: Known lower bound for gap calculation (optional)
            save_best_model: Whether to save the best model based on evaluation metric
            save_dir: Directory to save best model (required if save_best_model=True)
            metric_mode: "min" to minimize metric (e.g., makespan) or "max" to maximize (e.g., reward)
            metric_key: Key of the metric to track for best model (e.g., "makespan", "avg_return")
            cfg: Configuration object to check environment type (optional)
            model_factory: Function to create a fresh base model (for LoRA weight transfer)
        """
        super().__init__(
            env=env,
            logger=logger,
            max_steps=max_steps,
            lower_bound=lower_bound,
            save_best_model=save_best_model,
            save_dir=save_dir,
            metric_mode=metric_mode,
            metric_key=metric_key,
        )
        self.cfg = cfg
        self.is_aec_env = self._check_aec_environment()

    def _check_aec_environment(self) -> bool:
        """
        Check if the environment is an AEC (Agent Environment Cycle) type.

        Returns:
            True if the environment is AEC type, False otherwise
        """
        if self.cfg is not None and hasattr(self.cfg, "marl"):
            env_type = self.cfg.marl.get("env_type", None)
            return env_type == "aec"
        return False

    def transform_td(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Transform tensordict for AEC environments.

        This function filters out duplicate observations and restructures the tensordict
        to match the expected format for multi-agent evaluation. It assumes that:
        - Odd indices (1, 3, 5, ...) contain AGV-job decisions
        - Even indices (0, 2, 4, ...) contain job-job decisions
        - We keep only the odd indices (AGV-job decisions)

        Args:
            tensordict: The raw tensordict from rollout

        Returns:
            Transformed tensordict with filtered data
        """
        agv_selector_next_obs_key = ("next", "agv_selector", "observation")

        all_indices_of_agv_job = np.arange(
            tensordict[agv_selector_next_obs_key].shape[0]
        )[1::2]
        all_indices_of_job_job = np.arange(
            tensordict[agv_selector_next_obs_key].shape[0]
        )[0::2]

        for action_key in self.ACTION_RELATED_KEYS:
            all_action_of_job_job = tensordict[("job_selector", action_key)][
                all_indices_of_job_job
            ]
            tensordict[("job_selector", action_key)][all_indices_of_agv_job] = (
                all_action_of_job_job
            )

        new_tensorDict = tensordict[1::2]
        return new_tensorDict

    def evaluate(self, policy_module, step: int) -> dict[str, float]:
        """
        Run evaluation with multi-agent group structure.

        Args:
            policy_module: The policy to evaluate
            step: Current training step

        Returns:
            Dictionary of evaluation metrics aggregated across all agent groups
        """
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = self.env.rollout(
                self.max_steps,
                policy_module,
            )

            if self.is_aec_env:
                eval_rollout = self.transform_td(eval_rollout)

            eval_mask = eval_rollout["next", "done"]
            eval_terminated = eval_rollout["next", "terminated"]

            all_metrics = {}

            for group in self.env.group_map.keys():
                group_metrics = {
                    "mean_reward": eval_rollout["next", group, "reward"].mean().item(),
                    "sum_reward": eval_rollout["next", group, "reward"].sum().item(),
                    "step_count": eval_rollout["step_count"].max().item(),
                    "avg_return": eval_rollout["next", group, "episode_reward"][
                        eval_mask
                    ]
                    .mean()
                    .item(),
                    "makespan": eval_rollout["next", group, "info", "makespan"][
                        eval_terminated
                    ].item(),
                }

                self.logger.log_evaluation_metrics(
                    mean_reward=group_metrics["mean_reward"],
                    sum_reward=group_metrics["sum_reward"],
                    step_count=group_metrics["step_count"],
                    avg_return=group_metrics["avg_return"],
                    makespan=group_metrics["makespan"],
                    step=step,
                )

                if self.lower_bound is not None:
                    gap_to_lb = (
                        group_metrics["makespan"] - self.lower_bound
                    ) / self.lower_bound
                    self.logger.log_custom_metrics(
                        {f"eval/{group}/gap_to_lb": gap_to_lb}, step=step
                    )
                    group_metrics["gap_to_lb"] = gap_to_lb

                all_metrics[group] = group_metrics
                break

            first_group = list(self.env.group_map.keys())[0]
            primary_makespan = all_metrics[first_group]["makespan"]

            aggregated_metrics = {
                "makespan": primary_makespan,
                "step_count": all_metrics[first_group]["step_count"],
            }

            total_mean_reward = sum(m["mean_reward"] for m in all_metrics.values())
            total_sum_reward = sum(m["sum_reward"] for m in all_metrics.values())
            aggregated_metrics["mean_reward"] = total_mean_reward / len(all_metrics)
            aggregated_metrics["sum_reward"] = total_sum_reward

            if self.save_best_model:
                current_metric = aggregated_metrics.get(self.metric_key)
                if current_metric is not None:
                    is_best = False
                    if self.metric_mode == "min":
                        is_best = current_metric < self.best_metric
                    else:
                        is_best = current_metric > self.best_metric

                    if is_best:
                        self.best_metric = current_metric
                        self.best_step = step

                        lora_params = count_lora_parameters(policy_module)
                        if lora_params > 0:
                            policy_module_copy = copy.deepcopy(policy_module)
                            merge_all_lora_weights(policy_module_copy)

                            self._save_model(policy_module_copy, step, current_metric)
                        else:
                            policy_module_copy = copy.deepcopy(policy_module)
                            self._save_model(policy_module_copy, step, current_metric)

                        self.logger.log_custom_metrics(
                            {f"eval/best_{self.metric_key}": self.best_metric},
                            step=step,
                        )
                        print(
                            f"New best {self.metric_key}: {self.best_metric:.4f} at step {step}"
                        )

            del eval_rollout

            aggregated_metrics["groups"] = all_metrics
            return aggregated_metrics


class MultiAgentAdvantageEvaluator(MultiAgentEvaluator):
    """
    Multi-agent evaluator for advantage-based training with shared reward structure.

    This evaluator inherits from MultiAgentEvaluator and overrides only the reward
    extraction logic. The key difference from MultiAgentEvaluator is:
    - Rewards are accessed from eval_rollout["next", "reward"] (global, not per-group)
    - Makespan is still accessed per-group: eval_rollout["next", group, "info", "makespan"]

    For AEC (Agent Environment Cycle) environments, it automatically applies the
    transform_td function inherited from MultiAgentEvaluator.
    """

    def evaluate(self, policy_module, step: int) -> dict[str, float]:
        """
        Run evaluation with multi-agent advantage-based structure.

        In advantage-based training, rewards are global and shared across all agents,
        so we extract them from eval_rollout["next", "reward"] rather than per-group.

        Args:
            policy_module: The policy to evaluate
            step: Current training step

        Returns:
            Dictionary of evaluation metrics aggregated across all agent groups
        """
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = self.env.rollout(
                self.max_steps,
                policy_module,
            )

            if self.is_aec_env:
                eval_rollout = self.transform_td(eval_rollout)

            eval_mask = eval_rollout["next", "done"]
            eval_terminated = eval_rollout["next", "terminated"]

            reward = eval_rollout["next", "reward"]

            all_metrics = {}

            for group in self.env.group_map.keys():
                group_metrics = {
                    "mean_reward": reward.mean().item(),
                    "sum_reward": reward.sum().item(),
                    "step_count": eval_rollout["step_count"].max().item(),
                    "avg_return": reward[eval_mask].mean().item(),
                    "makespan": eval_rollout["next", group, "info", "makespan"][
                        eval_terminated
                    ].item(),
                }

                self.logger.log_evaluation_metrics(
                    mean_reward=group_metrics["mean_reward"],
                    sum_reward=group_metrics["sum_reward"],
                    step_count=group_metrics["step_count"],
                    avg_return=group_metrics["avg_return"],
                    makespan=group_metrics["makespan"],
                    step=step,
                    agent=group,
                )

                if self.lower_bound is not None:
                    gap_to_lb = (
                        group_metrics["makespan"] - self.lower_bound
                    ) / self.lower_bound
                    self.logger.log_custom_metrics(
                        {f"eval/{group}/gap_to_lb": gap_to_lb}, step=step
                    )
                    group_metrics["gap_to_lb"] = gap_to_lb

                all_metrics[group] = group_metrics

            first_group = list(self.env.group_map.keys())[0]
            primary_makespan = all_metrics[first_group]["makespan"]

            aggregated_metrics = {
                "makespan": primary_makespan,
                "step_count": all_metrics[first_group]["step_count"],
                "mean_reward": reward.mean().item(),
                "sum_reward": reward.sum().item(),
                "avg_return": reward[eval_mask].mean().item(),
            }

            if self.save_best_model:
                current_metric = aggregated_metrics.get(self.metric_key)
                if current_metric is not None:
                    is_best = False
                    if self.metric_mode == "min":
                        is_best = current_metric < self.best_metric
                    else:
                        is_best = current_metric > self.best_metric

                    if is_best:
                        self.best_metric = current_metric
                        self.best_step = step
                        self._save_model(policy_module, step, current_metric)

                        self.logger.log_custom_metrics(
                            {f"eval/best_{self.metric_key}": self.best_metric},
                            step=step,
                        )
                        print(
                            f"New best {self.metric_key}: {self.best_metric:.4f} at step {step}"
                        )

            del eval_rollout

            aggregated_metrics["groups"] = all_metrics
            return aggregated_metrics


def _list_agvs(instance):
    storage = {
        "jssp_instances/transport/ft06": [
            2,
            4,
            6,
        ],
        "jssp_instances/transport/ft10": [
            3,
            6,
            9,
        ],
        "jssp_instances/transport/ft20": [
            3,
            10,
            18,
        ],  # 20x5 ,
        "jssp_instances/transport/la24": [
            3,
            7,
            14,
        ],  # 15x10
        "jssp_instances/transport/la23": [
            3,
            7,
            14,
        ],  # 15x10
        "jssp_instances/transport/la15": [
            3,
            10,
            18,
        ],  # 20x5
        "jssp_instances/transport/la32": [
            3,
            15,
            25,
        ],
    }
    return storage[instance]


def create_agv_evaluator(
    env: TransformedEnv,
    logger: TrainingLogger,
    cfg: DictConfig,
    dispatcher: DispatcherBase = None,
) -> BaseEvaluator:
    """
    Factory function to create evaluator based on configuration.

    Args:
        env: The environment to evaluate on
        logger: Logger for recording evaluation metrics
        cfg: Configuration containing evaluation settings
            evaluation:
              type: "standard", "benchmark",, "multi_agent", "multi_agent_advantage", or "none"
              max_steps: int (default: 1000)
              lower_bound: float (optional, for standard evaluator)
              save_best_model: bool (default: False)
              save_dir: str (optional, required if save_best_model=True)
              metric_mode: "min" or "max" (default: "min")
              metric_key: str (default: "makespan")
              # For benchmark evaluator:
              use_generator: bool (default: False) - whether to use instance generator
              use_heuristic_reference: bool (default: False) - whether to use on-the-fly heuristic as reference
              generator_type: str (default: "random_uniform") - type of generator if use_generator=True
              generator_kwargs: dict (optional) - kwargs for generator
              reference_makespan: float (optional) - fixed reference makespan for generator-based evaluation
              heuristic_name: str (default: "mwr") - heuristic to use for on-the-fly reference computation
              dataset_filename: str (default: "jssp_instances/heuristic_solutions/solved_instances.json") - for dataset-based evaluation
              n_instances: int (default: 100)
              benchmark_heuristic: str (default: "mwkr") - for dataset-based evaluation with pre-computed solutions

    Returns:
        Configured evaluator instance
    """
    eval_type = cfg.evaluation.get("type", "standard")
    if eval_type == "multi_agent":
        return MultiAgentEvaluator(
            env=env,
            logger=logger,
            max_steps=cfg.evaluation.get("max_steps", 1000),
            lower_bound=cfg.env.get("lower_bound", None),
            save_best_model=cfg.evaluation.get("save_best_model", False),
            save_dir=cfg.get("save_dir", None),
            metric_mode=cfg.evaluation.get("metric_mode", "min"),
            metric_key=cfg.evaluation.get("metric_key", "makespan"),
            cfg=cfg,
        )

    elif eval_type == "benchmark_multi_agent":
        return MultiAgentBenchmark(
            env=env,
            logger=logger,
            max_steps=cfg.evaluation.get("max_steps", 1000),
            save_best_model=cfg.evaluation.save_best_model,
            save_dir=cfg.get("save_dir", None),
            metric_mode=cfg.evaluation.metric_mode,
            metric_key=cfg.evaluation.metric_key,
            cfg=cfg,
            dispatcher=dispatcher,
        )
    elif eval_type == "multi_agent_advantage":
        return MultiAgentAdvantageEvaluator(
            env=env,
            logger=logger,
            max_steps=cfg.evaluation.get("max_steps", 1000),
            lower_bound=cfg.env.get("lower_bound", None),
            save_best_model=cfg.evaluation.get("save_best_model", False),
            save_dir=cfg.get("save_dir", None),
            metric_mode=cfg.evaluation.get("metric_mode", "min"),
            metric_key=cfg.evaluation.get("metric_key", "makespan"),
            cfg=cfg,
        )

    return create_evaluator(env, logger, cfg, dispatcher)
