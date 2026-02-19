import torch
from omegaconf import DictConfig

from jssp_agv.dispatcher.mlp_dispatcher import MLPDispatcher
from jssp_agv.dispatcher.utils_gnn import (
    create_gnn_critic,
    create_gnn_environment,
    create_gnn_loss_modules,
    create_gnn_policies,
)
from jssp_agv.modules.lora import (
    apply_lora_to_gnn,
    apply_lora_to_linear_layers,
    freeze_non_lora_parameters,
)


class GNNDispatcher(MLPDispatcher):
    """Dispatcher for GNN-based training with standard GAE."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.loaded_external_job_policy = False
        self.loaded_external_agv_policy = False

    def setup_environment(self):
        """Create the GNN-based environment."""
        self.env = create_gnn_environment(self.cfg, self.instance)
        return self.env

    def loaded_job_policy_differs(self) -> bool:
        """Check whether parameters of the current job_selector differ from loaded_job_policy.

        Returns:
            True if they differ (including different keys/shapes or values), False if identical.

        Raises:
            RuntimeError: if loaded_job_policy is not set or job_selector not found.
        """
        if getattr(self, "loaded_job_policy", None) is None:
            self.loaded_job_policy = (
                self.combined_policies["job_selector"].module[0].module
            )
            print(
                "Warning: No loaded_job_policy is set. Call set_loaded_pre_trained_job_policy(...) first."
                "Now setting loaded_job_policy to current job_selector for comparison."
            )

        current_model = self.combined_policies["job_selector"].module[0].module
        loaded_model = self.loaded_job_policy

        # Get state dicts
        current_sd = current_model.state_dict()
        loaded_sd = loaded_model.state_dict()

        # Quick check: key sets
        current_keys = set(current_sd.keys())
        loaded_keys = set(loaded_sd.keys())

        # Compare each parameter tensor
        for k in current_sd.keys():
            cur_tensor = current_sd[k]
            load_tensor = loaded_sd[k]

            # Use allclose to allow for floating point tolerance
            if not torch.allclose(cur_tensor, load_tensor, rtol=1e-5, atol=1e-8):
                return True

        # All parameters appear identical
        return False

    def loaded_agv_policy_differs(self) -> bool:
        """Check whether parameters of the current job_selector differ from loaded_job_policy.

        Returns:
            True if they differ (including different keys/shapes or values), False if identical.

        Raises:
            RuntimeError: if loaded_agv_policy is not set or job_selector not found.
        """
        if getattr(self, "loaded_agv_policy", None) is None:
            self.loaded_agv_policy = self.combined_policies["agv_selector"][0].module
            print(
                "Warning: No loaded_agv_policy is set. Call set_loaded_pre_trained_agv_policy(...) first."
                "Now setting loaded_agv_policy to current agv_selector for comparison."
            )

        current_model = self.combined_policies["agv_selector"][0].module
        loaded_model = self.loaded_agv_policy

        current_sd = current_model.state_dict()
        loaded_sd = loaded_model.state_dict()

        current_keys = set(current_sd.keys())
        loaded_keys = set(loaded_sd.keys())

        for k in current_sd.keys():
            cur_tensor = current_sd[k]
            load_tensor = loaded_sd[k]

            if not torch.allclose(cur_tensor, load_tensor, rtol=1e-5, atol=1e-8):
                return True

        return False

    def setup_models(self):
        """Create GNN policies and critic with shared feature extractor."""
        self.policies, self.combined_policies, self.shared_extractor = (
            create_gnn_policies(self.env, self.cfg)
        )
        self.value_module = create_gnn_critic(self.env, self.shared_extractor, self.cfg)
        self._check_for_job_selector_finetune()
        self._check_for_agv_selector_finetune()

        return (self.policies, self.combined_policies), self.value_module

    def _check_for_job_selector_finetune(self):
        job_selector_attr = hasattr(self.cfg.finetune, "job_selector")
        if not job_selector_attr:
            apply_finetune = self.cfg.finetune.finetune or self.cfg.finetune.lora
        else:
            apply_finetune = self.cfg.finetune.job_selector

        if apply_finetune:
            # print("Job Selector fine-tuning specified in config.")
            job_selector_kwargs_attr = hasattr(self.cfg.finetune, "job_selector_kwargs")
            if not job_selector_kwargs_attr:
                job_cfg = self.cfg.finetune
            else:
                job_cfg = self.cfg.finetune.job_selector_kwargs
            if job_cfg.lora or job_cfg.finetune:
                print("=== Replacing job_selector with loaded pre-trained policy ===")
                dif_before = self.loaded_job_policy_differs()
                self.combined_policies["job_selector"].module[0].module.load_state_dict(
                    self.loaded_job_policy.state_dict()
                )

                dif_after = self.loaded_job_policy_differs()
                if not dif_before and self.loaded_external_job_policy:
                    raise RuntimeError(
                        "Expected loaded_job_policy_differs() to be True before loading the state dict, but it was False."
                    )

                if dif_after and self.loaded_external_job_policy:
                    raise RuntimeError(
                        "Expected loaded_job_policy_differs() to be False after loading the state dict, but it was True."
                    )

                del self.loaded_job_policy
                torch.cuda.empty_cache()
                if job_cfg.lora:
                    # self.shared_extractor = self.combined_policies["job_selector"].module[0].shared_extractor
                    # self.load_model_state_dict(self.policies["job_selector"], job_cfg.policy_path)
                    # self.combined_policies["job_selector"].module[0].module.policy_head.actor_policy_head[0].weight
                    self._collect_before_lora_stats_job()

                    print("=== Applying LoRA ===")
                    self.apply_lora_job()
                    self._collect_after_lora_stats_job()
                    self._summarize_lora_stats_job()

    def _check_for_agv_selector_finetune(self):
        job_selector_attr = hasattr(self.cfg.finetune, "agv_selector")
        if not job_selector_attr:
            return
        else:
            apply_finetune = self.cfg.finetune.agv_selector

        if apply_finetune:
            # print("AGV Selector fine-tuning specified in config.")
            agv_cfg = self.cfg.finetune.agv_selector_kwargs
            if agv_cfg.lora or agv_cfg.finetune:
                print("=== Replacing agv_selector with loaded pre-trained policy ===")
                dif_before = self.loaded_agv_policy_differs()
                self.combined_policies["agv_selector"][0].module.load_state_dict(
                    self.loaded_agv_policy.state_dict()
                )

                dif_after = self.loaded_agv_policy_differs()
                if not dif_before and self.loaded_external_agv_policy:
                    raise RuntimeError(
                        "Expected loaded_agv_policy_differs() to be True before loading the state dict, but it was False."
                    )

                if dif_after and self.loaded_external_agv_policy:
                    raise RuntimeError(
                        "Expected loaded_agv_policy_differs() to be False after loading the state dict, but it was True."
                    )

                del self.loaded_agv_policy
                torch.cuda.empty_cache()
                if agv_cfg.lora:
                    self._collect_before_lora_stats_agv()

                    print("=== Applying LoRA ===")
                    self.apply_lora_agv()
                    self._collect_after_lora_stats_agv()
                    self._summarize_lora_stats_agv()

    def setup_loss_modules(self):
        """Create PPO loss modules with standard GAE."""
        self.loss_modules = create_gnn_loss_modules(
            self.policies, self.value_module, self.cfg
        )
        return self.loss_modules, None

    def setup_test_environment(self):
        """Create the test environment."""
        test_env_cfg = self.cfg.copy()
        test_env_cfg["env"]["random_instance"] = (
            False  # Set to original instance for testing
        )
        self.test_env = create_gnn_environment(test_env_cfg, self.instance)
        return self.test_env

    def set_loaded_pre_trained_job_policy(self, job_selector):
        """Set the loaded pre-trained job selector policy."""
        self.loaded_job_policy = job_selector
        self.loaded_external_job_policy = True

    def set_loaded_pre_trained_agv_policy(self, agv_selector):
        """Set the loaded pre-trained job selector policy."""
        self.loaded_agv_policy = agv_selector
        self.loaded_external_agv_policy = True

    def apply_lora_job(self):
        """Apply LoRA to GNN and Policy Head if specified in config."""
        job_cfg = self.cfg.finetune.job_selector_kwargs
        lora_gnn = job_cfg.gnn
        lora_pol_head = job_cfg.policy_head
        if lora_gnn:
            self.combined_policies["job_selector"].module[
                0
            ].shared_extractor.gnn = apply_lora_to_gnn(
                gnn_encoder=self.combined_policies["job_selector"]
                .module[0]
                .shared_extractor.gnn,
                rank=job_cfg.rank,
                alpha=job_cfg.alpha,
                dropout=job_cfg.dropout,
            )

        if lora_pol_head:
            self.combined_policies["job_selector"].module[
                0
            ].policy_head = apply_lora_to_linear_layers(
                model=self.combined_policies["job_selector"].module[0].policy_head,
                rank=job_cfg.rank,
                alpha=job_cfg.alpha,
                dropout=job_cfg.dropout,
                target_modules=["policy_head"],
            )

        # How to check the freeze parameters?
        freeze_non_lora_parameters(self.combined_policies["job_selector"].module[0])

        self._check_lora_models("job_selector")

    def apply_lora_agv(self):
        """Apply LoRA to GNN and Policy Head if specified in config."""
        agv_cfg = self.cfg.finetune.agv_selector_kwargs

        self.combined_policies["agv_selector"][
            0
        ].module.policy_head = apply_lora_to_linear_layers(
            model=self.combined_policies["agv_selector"][0].module.policy_head,
            rank=agv_cfg.rank,
            alpha=agv_cfg.alpha,
            dropout=agv_cfg.dropout,
            target_modules=["policy_head"],
        )

        # How to check the freeze parameters?
        freeze_non_lora_parameters(self.combined_policies["agv_selector"].module[0])

        self._check_lora_models("agv_selector")

    def _collect_before_lora_stats_job(self):
        self.job_selector_total_params_before = sum(
            p.numel() for p in self.combined_policies["job_selector"].parameters()
        )
        self.job_selector_trainable_params_before = sum(
            p.numel()
            for p in self.combined_policies["job_selector"].parameters()
            if p.requires_grad
        )

        self.total_params_before = sum(
            p.numel() for p in self.combined_policies.parameters()
        )
        self.total_trainable_params_before = sum(
            p.numel() for p in self.combined_policies.parameters() if p.requires_grad
        )

    def _collect_before_lora_stats_agv(self):
        self.agv_selector_total_params_before = sum(
            p.numel() for p in self.combined_policies["agv_selector"].parameters()
        )
        self.agv_selector_trainable_params_before = sum(
            p.numel()
            for p in self.combined_policies["agv_selector"].parameters()
            if p.requires_grad
        )

        self.total_params_before = sum(
            p.numel() for p in self.combined_policies.parameters()
        )
        self.total_trainable_params_before = sum(
            p.numel() for p in self.combined_policies.parameters() if p.requires_grad
        )

    def _collect_after_lora_stats_job(self):
        self.job_selector_total_params_after = sum(
            p.numel() for p in self.combined_policies["job_selector"].parameters()
        )
        self.job_selector_trainable_params_after = sum(
            p.numel()
            for p in self.combined_policies["job_selector"].parameters()
            if p.requires_grad
        )
        self.total_params_after = sum(
            p.numel() for p in self.combined_policies.parameters()
        )
        self.total_trainable_params_after = sum(
            p.numel() for p in self.combined_policies.parameters() if p.requires_grad
        )

    def _collect_after_lora_stats_agv(self):
        self.agv_selector_total_params_after = sum(
            p.numel() for p in self.combined_policies["agv_selector"].parameters()
        )
        self.agv_selector_trainable_params_after = sum(
            p.numel()
            for p in self.combined_policies["agv_selector"].parameters()
            if p.requires_grad
        )

        self.total_params_after = sum(
            p.numel() for p in self.combined_policies.parameters()
        )
        self.total_trainable_params_after = sum(
            p.numel() for p in self.combined_policies.parameters() if p.requires_grad
        )

    def _summarize_lora_stats_agv(self):
        print("\n" + "=" * 80)
        print("LoRA Parameter Summary".center(80))
        print("=" * 80)

        # Calculate ratios
        agv_before_ratio = (
            (
                self.agv_selector_trainable_params_before
                / self.agv_selector_total_params_before
                * 100
            )
            if self.agv_selector_total_params_before
            else 0
        )
        agv_after_ratio = (
            (
                self.agv_selector_trainable_params_after
                / self.agv_selector_total_params_after
                * 100
            )
            if self.agv_selector_total_params_after
            else 0
        )
        total_before_ratio = (
            (self.total_trainable_params_before / self.total_params_before * 100)
            if self.total_params_before
            else 0
        )
        total_after_ratio = (
            (self.total_trainable_params_after / self.total_params_after * 100)
            if self.total_params_after
            else 0
        )

        # Print table header
        print(f"{'Model':<20} {'Metric':<20} {'Before LoRA':>18} {'After LoRA':>18}")
        print("-" * 80)

        # Job Selector rows
        print(
            f"{'AGV Selector':<20} {'Total Params':<20} {self.agv_selector_total_params_before:>18,} {self.agv_selector_total_params_after:>18,}"
        )
        print(
            f"{'':20} {'Trainable Params':<20} {self.agv_selector_trainable_params_before:>18,} {self.agv_selector_trainable_params_after:>18,}"
        )
        print(
            f"{'':20} {'Trainable Ratio':<20} {agv_before_ratio:>17.2f}% {agv_after_ratio:>17.2f}%"
        )

        print("-" * 80)

        # Total (All Policies) rows
        print(
            f"{'All Policies':<20} {'Total Params':<20} {self.total_params_before:>18,} {self.total_params_after:>18,}"
        )
        print(
            f"{'':20} {'Trainable Params':<20} {self.total_trainable_params_before:>18,} {self.total_trainable_params_after:>18,}"
        )
        print(
            f"{'':20} {'Trainable Ratio':<20} {total_before_ratio:>17.2f}% {total_after_ratio:>17.2f}%"
        )

        print("=" * 80 + "\n")

    def _summarize_lora_stats_job(self):
        print("\n" + "=" * 80)
        print("LoRA Parameter Summary".center(80))
        print("=" * 80)

        # Calculate ratios
        job_before_ratio = (
            (
                self.job_selector_trainable_params_before
                / self.job_selector_total_params_before
                * 100
            )
            if self.job_selector_total_params_before
            else 0
        )
        job_after_ratio = (
            (
                self.job_selector_trainable_params_after
                / self.job_selector_total_params_after
                * 100
            )
            if self.job_selector_total_params_after
            else 0
        )
        total_before_ratio = (
            (self.total_trainable_params_before / self.total_params_before * 100)
            if self.total_params_before
            else 0
        )
        total_after_ratio = (
            (self.total_trainable_params_after / self.total_params_after * 100)
            if self.total_params_after
            else 0
        )

        # Print table header
        print(f"{'Model':<20} {'Metric':<20} {'Before LoRA':>18} {'After LoRA':>18}")
        print("-" * 80)

        # Job Selector rows
        print(
            f"{'Job Selector':<20} {'Total Params':<20} {self.job_selector_total_params_before:>18,} {self.job_selector_total_params_after:>18,}"
        )
        print(
            f"{'':20} {'Trainable Params':<20} {self.job_selector_trainable_params_before:>18,} {self.job_selector_trainable_params_after:>18,}"
        )
        print(
            f"{'':20} {'Trainable Ratio':<20} {job_before_ratio:>17.2f}% {job_after_ratio:>17.2f}%"
        )

        print("-" * 80)

        # Total (All Policies) rows
        print(
            f"{'All Policies':<20} {'Total Params':<20} {self.total_params_before:>18,} {self.total_params_after:>18,}"
        )
        print(
            f"{'':20} {'Trainable Params':<20} {self.total_trainable_params_before:>18,} {self.total_trainable_params_after:>18,}"
        )
        print(
            f"{'':20} {'Trainable Ratio':<20} {total_before_ratio:>17.2f}% {total_after_ratio:>17.2f}%"
        )

        print("=" * 80 + "\n")

    def _check_lora_models(self, agent_name):
        assert self.combined_policies[agent_name] == self.policies[agent_name]
