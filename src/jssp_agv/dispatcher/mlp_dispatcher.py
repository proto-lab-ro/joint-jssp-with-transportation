from omegaconf import DictConfig

from jssp_agv.dispatcher.utils_mlp import (
    create_collectors,
    create_critics,
    create_loss_modules,
    create_optimizers,
    create_policy,
    create_replay_buffer,
    create_schedulers,
    get_instance,
)
from jssp_core.domain.base_dispatcher import DispatcherBase


class MLPDispatcher(DispatcherBase):
    """Dispatcher for MLP-based training with standard setup."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def setup_instance(self):
        """Load the problem instance."""
        self.instance = get_instance(self.cfg)
        return self.instance

    def setup_environment(self):
        """Create the standard MLP-based environment."""
        # self.env = create_environment(self.cfg, self.instance)
        return None

    def setup_test_environment(self):
        return None

    def setup_models(self):
        """Create MLP policies and critic."""
        self.policies, self.combined_policies = create_policy(self.env, self.cfg)
        self.value_module = create_critics(self.env, self.cfg)
        return (self.policies, self.combined_policies), self.value_module

    def setup_loss_modules(self):
        """Create PPO loss modules with integrated GAE."""
        self.loss_modules = create_loss_modules(
            self.policies, self.value_module, self.cfg
        )
        return self.loss_modules, None

    def setup_optimizers(self):
        """Create optimizers for each agent."""
        self.optimizers = create_optimizers(self.loss_modules, self.cfg)
        return self.optimizers

    def setup_schedulers(self):
        """Create learning rate schedulers."""
        self.schedulers = create_schedulers(self.optimizers, self.cfg)
        return self.schedulers

    def setup_collector(self):
        """Create data collector for experience gathering."""
        self.collector = create_collectors(self.env, self.combined_policies, self.cfg)
        return self.collector

    def setup_replay_buffer(self):
        """Create replay buffers for each agent group."""
        self.replay_buffers = create_replay_buffer(self.env, self.cfg)
        return self.replay_buffers
