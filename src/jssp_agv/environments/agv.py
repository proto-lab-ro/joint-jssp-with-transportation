import numpy as np
import torch

from jssp_core.domain import ObservationType
from jssp_core.environments import JSSP_AGV, JsspAGVEnvAEC


class GnnJssp_AGVEnv(JSSP_AGV):
    def __init__(self, instance, **kwargs):
        super().__init__(instance=instance, **kwargs)

        self.observation_type = self.observation_provider.observation_type

    def reset(self, seed: int | None = None, options: dict | None = None):
        (observation_dict, info_dict) = super().reset(seed=seed, options=options)

        if self.observation_type in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            if isinstance(observation_dict["job_selector_0"], dict):
                graph_observation_dict = {
                    "job_selector_0": {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in observation_dict["job_selector_0"].items()
                    },
                }
            else:
                graph_observation_dict = observation_dict
        else:
            graph_observation_dict = observation_dict

        return graph_observation_dict, info_dict

    def step(self, action_dict: dict):
        op_action = action_dict["job_selector_0"].item()  # Flatt Action?

        # job_action = self.schedule.flat_index_to_job_op(op_action)[0]

        actions = {
            "job_selector_0": op_action,
        }

        (
            observation_dict,
            reward_dict,
            terminated_dict,
            truncated_dict,
            info_dict,
        ) = super().step(actions)

        if self.observation_type in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            if isinstance(observation_dict["job_selector_0"], dict):
                graph_observation_dict = {
                    "job_selector_0": {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in observation_dict["job_selector_0"].items()
                    },
                }
            else:
                graph_observation_dict = observation_dict
        else:
            graph_observation_dict = observation_dict

        return (
            graph_observation_dict,
            reward_dict,
            terminated_dict,
            truncated_dict,
            info_dict,
        )


class GnnJsspAGVEnvAEC(JsspAGVEnvAEC):
    """
    A Transprt JJSP environment with GNN-based observations from the Parallel
    PettingZoo Environment.
    This class acts only as an wrapper and uses Graph Observations for the Job Scheduler.
    """

    def __init__(self, instance, **kwargs):
        super().__init__(instance=instance, **kwargs)
        self.observation_type = self.get_observation_type("job_selector_0")

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)

        if self.observation_type in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            if isinstance(self.observations["job_selector_0"], dict):
                self.observations["job_selector_0"] = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in self.observations["job_selector_0"].items()
                }

    def step(self, action: int):
        super().step(action)
        if self.observation_type in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            if isinstance(self.observations["job_selector_0"], dict):
                self.observations["job_selector_0"] = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in self.observations["job_selector_0"].items()
                }
