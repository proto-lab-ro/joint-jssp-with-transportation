"""
Reward functions for JSSP environment.
Allows easy experimentation with different reward strategies.
"""

from typing import Any

from jssp_core.domain.reward import RewardFunction
from jssp_core.registry import REWARD_REGISTRY
from jssp_core.schedule import Schedule, TransportSchedule


@REWARD_REGISTRY.register("negative_sparse_makespan")
class NegativeSparseMakespanReward(RewardFunction):
    """
    Sparse reward function that only provides a reward at the end of the episode
    based on the makespan of the final schedule.
    ️   R = - (makespan / lower_bound)
    ️   A lower makespan results in a less negative reward.
    ️   scaling_factor scales the magnitude of the reward.
    """

    def __init__(
        self, initial_schedule: Schedule, scaling_factor: float = 100, **kwargs
    ):
        super().__init__(initial_schedule, **kwargs)

        self.lb = initial_schedule.get_lower_bound_makespan()
        self.scaling_factor = scaling_factor

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule: Schedule = reward_data["schedule"]
        reward = 0.0
        if schedule.is_complete():
            reward = -schedule.get_makespan() / (self.lb * self.scaling_factor)
        return reward

    @property
    def name(self) -> str:
        return "negative_sparse_makespan"


# Registry of available reward functions (deprecated, use REWARD_REGISTRY)
REWARD_FUNCTIONS = REWARD_REGISTRY._registry


def get_reward_function(
    name: str, initial_schedule: Schedule, **kwargs
) -> RewardFunction:
    """
    Factory function to create reward functions

    Args:
        name: Name of the reward function
        **kwargs: Arguments to pass to the reward function constructor

    Returns:
        RewardFunction instance
    """
    reward_cls = REWARD_REGISTRY.get(name)

    if isinstance(initial_schedule, TransportSchedule) and name in [
        "makespan_improvement",
        "dense_shaped",
        "lower_bound_makespan",
        "sparse_makespan",
        "sparse_exponential",
        "max_operation_lower_bound_difference",
    ]:
        # print(
        #     f"Warning: Reward function '{name}' may not be fully compatible with TransportSchedule."
        # )
        return reward_cls(initial_schedule=initial_schedule, **kwargs)

    return reward_cls(initial_schedule=initial_schedule, **kwargs)


def list_reward_functions() -> dict[str, str]:
    """Return a dict of available reward functions and their descriptions"""
    return {name: name for name in REWARD_REGISTRY.list_available()}
