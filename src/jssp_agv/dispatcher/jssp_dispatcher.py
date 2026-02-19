from omegaconf import DictConfig
from torch import nn

from jssp_agv.dispatcher.mlp_dispatcher import MLPDispatcher
from jssp_agv.dispatcher.utils_gnn import (
    create_gnn_critic,
    create_gnn_loss_modules,
    create_gnn_policies,
)
from jssp_agv.dispatcher.utils_jssp import (
    create_environment,
)
from jssp_core.solver import Heuristic


class JsspDispatcher(MLPDispatcher):
    """Dispatcher for MLP-based training with standard setup."""

    def __init__(self, cfg: DictConfig, agv_selector: nn.Module | Heuristic):
        self.agv_selector = agv_selector
        self.cfg = cfg
        self.instance = None
        self.env = None
        self.policies = None
        self.combined_policies = None
        self.value_module = None
        self.loss_modules = None
        self.optimizers = None
        self.schedulers = None
        self.collector = None
        self.replay_buffers = None

    def setup_environment(self):
        """Create the standard MLP-based environment."""
        self.env = create_environment(self.cfg, self.instance, self.agv_selector)
        return self.env

    def setup_test_environment(self):
        """Create the test environment."""
        test_env_cfg = self.cfg.copy()
        test_env_cfg["env"]["random_instance"] = (
            False  # Set to original instance for testing
        )
        self.test_env = create_environment(
            test_env_cfg, self.instance, self.agv_selector
        )
        return self.test_env

    def setup_models(self):
        """Create GNN policies and critic with shared feature extractor."""

        self.policies, self.combined_policies, self.shared_extractor = (
            create_gnn_policies(self.env, self.cfg)
        )
        self.value_module = create_gnn_critic(self.env, self.shared_extractor, self.cfg)

        return (self.policies, self.combined_policies), self.value_module

    def setup_loss_modules(self):
        """Create PPO loss modules with standard GAE."""

        self.loss_modules = create_gnn_loss_modules(
            self.policies, self.value_module, self.cfg
        )
        return self.loss_modules, None
