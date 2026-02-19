import functools

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv
from torch import nn

from jssp_core.domain import (
    AgvAgentType,
    EnvironmentType,
    JobSelectorType,
)
from jssp_core.instances import (
    InstanceGenerator,
    InstanceGeneratorLike,
    ensure_instance_generator,
)
from jssp_core.observation_providers import (
    ObservationProvider,
    get_observation_provider,
)
from jssp_core.observation_providers.agv import get_agv_observation_provider
from jssp_core.reward_functions import RewardFunction, get_reward_function
from jssp_core.schedule import Schedule, TransportSchedule
from jssp_core.solver import Heuristic


class JSSP_AGV(ParallelEnv):
    metadata = {"name": "jssp_with_agv"}

    def __init__(
        self,
        instance,
        max_episode_steps=2000,
        random_instance=True,
        reward_function: str | RewardFunction = "sparse_exponential",
        reward_kwargs: dict | None = None,
        agv_selector: nn.Module | Heuristic = None,
        agv_selector_kwargs: dict | None = None,
        job_observation_provider: str | ObservationProvider = "default",
        job_observation_kwargs: dict | None = None,
        number_agvs=3,
        render_mode=None,
        job_selector_type: JobSelectorType = JobSelectorType.OPERATION,
        instance_generator: InstanceGeneratorLike | None = None,
        instance_generator_kwargs: dict | None = None,
    ):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.agv_selector = agv_selector
        self.agv_selector_kwargs = agv_selector_kwargs or {}

        self._instance_generator: InstanceGenerator | None = None
        self._instance_generator_spec = instance_generator
        self._instance_generator_kwargs = dict(instance_generator_kwargs or {})
        self.possible_agents = ["job_selector_0"]
        self.agents = self.possible_agents.copy()

        self.job_selector_type = job_selector_type
        self.environment_type = EnvironmentType.SINGLE_AGENT

        self.agent_name_mapping = dict(
            zip(
                self.possible_agents,
                list(range(len(self.possible_agents))),
                strict=False,
            )
        )
        self.render_mode = None

        self.original_instance = instance
        self.random_instance = random_instance
        self.max_episode_steps = max_episode_steps
        self.reward_function = reward_function
        self.reward_kwargs = reward_kwargs or {}
        self.job_observation_kwargs = job_observation_kwargs or {}
        # self.agv_observation_kwargs = agv_observation_kwargs or {}

        # Initialize with the provided instance to get dimensions
        self.schedule = TransportSchedule(instance, num_agvs=number_agvs)

        if self.random_instance:
            self._instance_generator = ensure_instance_generator(
                self._instance_generator_spec,
                num_jobs=self.num_jobs,
                num_machines=self.num_machines - 2,
                generator_kwargs=self._instance_generator_kwargs,
            )

        self.set_up_agv_selector()
        self.set_up_reward_function(reward_kwargs)

        self.set_up_observation_provider(job_observation_provider)

        self.observation_types = {}
        for agent in self.possible_agents:
            if agent.__contains__("job"):
                self.observation_types[agent] = self.observation_provider

        self.step_count = 0
        self.initial_est_makespan = None

    @property
    def num_operations(self):
        return self.schedule.num_operations

    @property
    def num_jobs(self):
        return self.schedule.num_jobs

    @property
    def num_agvs(self):
        return self.schedule.num_agvs

    @num_agvs.setter
    def num_agvs(self, value):
        self.schedule.num_agvs = value

    @property
    def num_machines(self):
        return self.schedule.num_machines

    @property
    def agvs(self):
        return self.schedule.agvs

    @property
    def instance(self):
        return self.schedule.instance

    @instance.setter
    def instance(self, new_instance):
        """Set a new instance and reinitialize the schedule"""
        self.original_instance = new_instance
        self.schedule = TransportSchedule(new_instance, self.num_agvs)

    def get_makespan(self):
        return self.schedule.get_makespan()

    def set_up_agv_selector(self):
        if isinstance(self.agv_selector, Heuristic):
            self.agv_agent_type = AgvAgentType.HEURISTIC
        elif isinstance(self.agv_selector, nn.Module):
            self.agv_agent_type = AgvAgentType.POLICY

        else:
            raise ValueError("agv_selector must be Heuristic or nn.Module instance")
        self.set_up_agv_observation_provider()

    def set_up_agv_observation_provider(self):
        agv_observation_provider = self.agv_selector_kwargs["agv_observation_provider"]
        agv_observation_kwargs = self.agv_selector_kwargs["agv_observation_kwargs"]
        self.agv_policy_observation_type = agv_observation_kwargs["observation_type"]
        if isinstance(agv_observation_provider, str):
            self.agv_observation_provider = get_agv_observation_provider(
                agv_observation_provider, self.schedule, **agv_observation_kwargs
            )
        elif isinstance(agv_observation_provider, ObservationProvider):
            self.agv_observation_provider = agv_observation_provider
        else:
            raise ValueError(
                "agv_observation_provider must be a string name or ObservationProvider instance"
            )

    def set_up_reward_function(self, reward_kwargs):
        # Set up reward function
        if reward_kwargs is None:
            reward_kwargs = {}
        self.truncate_if_invalid = reward_kwargs.get("truncate_if_invalid", False)

        if isinstance(self.reward_function, str):
            self.reward_function = get_reward_function(
                self.reward_function,
                initial_schedule=TransportSchedule(
                    self.original_instance, num_agvs=self.num_agvs
                ),
                **reward_kwargs,
            )
        elif isinstance(self.reward_function, RewardFunction):
            self.reward_function = self.reward_function
        else:
            raise ValueError(
                "reward_function must be a string name or RewardFunction instance"
            )

    def set_up_observation_provider(self, job_observation_provider):
        # Set up observation provider
        if isinstance(job_observation_provider, str):
            self.observation_provider = get_observation_provider(
                job_observation_provider, self.schedule, **self.job_observation_kwargs
            )
        elif isinstance(job_observation_provider, ObservationProvider):
            self.observation_provider = job_observation_provider
        else:
            raise ValueError(
                "observation_provider must be a string name or ObservationProvider instance"
            )

    def _get_info(self):
        """Get additional info about the environment state"""
        # Get schedule summary and unpack it into separate entries
        schedule_summary = self.schedule.get_schedule_summary()
        current_makespan = self.get_makespan()
        job_selection_agent_info = {
            "step_count": self.step_count,
            "initial_est_makespan": (
                self.initial_est_makespan
                if self.initial_est_makespan is not None
                else 0
            ),
        }

        job_selection_agent_info.update(schedule_summary)

        info_dict = {}

        job_selection_agent_info = {
            "action_mask": self.action_mask_job_agent(),
            "makespan": current_makespan,
        }

        info_dict["job_selector_0"] = job_selection_agent_info
        return info_dict

    def _get_obs(self):
        """Get the current observation using the configured observation provider"""
        observation_dict = {}
        for agent, obs_provider in self.observation_types.items():
            agent_obs = obs_provider.get_observation(self.schedule)
            observation_dict[agent] = agent_obs
        return observation_dict

    def get_schedule_copy(self) -> Schedule:
        """Get a copy of the current schedule for external use"""
        return self.schedule.copy()

    def _validate_job_action(self, action_job):
        if not self.schedule.can_schedule_job(action_job):
            raise RuntimeError(f"Selected job cannot be scheduled: {action_job}")

        if not (0 <= action_job < self.num_jobs):
            raise ValueError(
                f"Invalid job index {action_job}, must be in [0, {self.num_jobs - 1}]"
            )

    def _validate_agv_action(self, action_agv):
        if not (0 <= action_agv < self.num_agvs):
            raise ValueError(
                f"Invalid agv index {action_agv}, must be in [0, {self.num_agvs - 1}]"
            )
        if action_agv not in self.agvs:
            raise ValueError(
                f"Invalid action_agv index {action_agv}, must be in {self.agvs}"
            )

    def _compare_instance_generator(self):
        """
        return True if instance_generator attributes (nunm_jobs, num_machines) are NOT the same as the currently defined instance
        """

        if (
            self._instance_generator.num_jobs != self.num_jobs
            or self._instance_generator.num_machines != self.num_machines - 2
        ):
            return True
        return False

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment to initial state"""

        self.agents = self.possible_agents
        self.step_count = 0

        # Generate new random instance if requested
        if self.random_instance:
            if self._instance_generator is None:
                self._instance_generator = ensure_instance_generator(
                    self._instance_generator_spec,
                    num_jobs=self.num_jobs,
                    num_machines=self.num_machines - 2,
                    generator_kwargs=self._instance_generator_kwargs,
                )

            if self._compare_instance_generator():
                self._instance_generator = ensure_instance_generator(
                    self._instance_generator_spec,
                    num_jobs=self.num_jobs,
                    num_machines=self.num_machines - 2,
                    generator_kwargs=self._instance_generator_kwargs,
                )
            new_instance = self._instance_generator.generate(seed=seed)
            self.schedule = TransportSchedule(new_instance, num_agvs=self.num_agvs)
            self.reward_function = get_reward_function(
                self.reward_function.name,
                initial_schedule=self.schedule,
                **self.reward_kwargs,
            )
        else:
            # Reset the schedule with original instance
            self.schedule = TransportSchedule(
                self.original_instance, num_agvs=self.num_agvs
            )

        # Reset observation provider with new schedule
        self.observation_provider.reset(self.schedule)
        self.agv_observation_provider.reset(self.schedule)
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, actions: dict):
        """
        Take a step in the environment by scheduling an operation

        Args:
            actions: action dict with entry for each agent with seelcted object
            For operation_scheduler: number of operation transformed to job_idx -> job_id
            For Agv_scheduler: Id of AGV to transport the selected operation

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """

        self.step_count += 1
        action_job = actions["job_selector_0"]

        if self.job_selector_type == JobSelectorType.OPERATION:
            action_job = self.schedule.flat_index_to_job_op(action_job)[0]
        # Check for episode timeout
        if self.step_count > self.max_episode_steps:
            reward_dict = {agent: 0.0 for agent in self.possible_agents}
            terminated_dict = {agent: False for agent in self.possible_agents}
            truncated_dict = {agent: True for agent in self.possible_agents}
            return (
                self._get_obs(),
                reward_dict,
                terminated_dict,
                truncated_dict,
                self._get_info(),
            )

        self._validate_job_action(action_job)
        self.set_selected_job_aec(
            action_job
        )  # Important for the observation_provider of the AGV if Policy
        action_agv = self._get_agv_scheduler_action(action_job)
        self._validate_agv_action(action_agv)

        state_before = self.schedule.copy()

        # Schedule the operation
        success = self.schedule.schedule_job(action_job, **{"agv": action_agv})

        if not success:
            # Calculate reward for failed scheduling
            # state_after_failed = self.schedule  # Current state after failed attempt
            raise ValueError(
                f"Tried to schedule non-permitted operation {action_job}, with AGV: {action_agv}, "
            )
            reward = -1  # Negative reward for invalid action
            if self.truncate_if_invalid:
                return self._get_obs(), reward, True, True, self._get_info()
            else:
                return self._get_obs(), reward, False, False, self._get_info()

        state_after = self.schedule

        # Calculate reward
        reward_dict = self._calculate_reward(state_before, state_after)

        # Check if episode is done
        terminated = self.schedule.is_complete()
        terminated_dict = {agent: terminated for agent in self.possible_agents}
        truncated_dict = {agent: False for agent in self.possible_agents}
        return (
            self._get_obs(),
            reward_dict,
            terminated_dict,
            truncated_dict,
            self._get_info(),
        )

    def _get_agv_scheduler_action(self, job_action):
        if self.agv_agent_type == AgvAgentType.HEURISTIC:
            return self.agv_selector.step(
                self.schedule, job_action
            )  # self.agv_selector
        elif self.agv_agent_type == AgvAgentType.POLICY:
            obs_tensor = self._get_agv_observation()

            with torch.no_grad():
                self.agv_selector.eval()
                agv_logits = self.agv_selector(obs_tensor).view(1, -1)
                # Get AGV action mask (all AGVs are typically valid)
                action_mask = np.ones(self.num_agvs, dtype=np.int8)

                # Apply action mask
                masked_logits = agv_logits.masked_fill(
                    ~torch.tensor(action_mask, dtype=torch.bool),
                    float("-inf"),
                )
                probabilities = torch.softmax(masked_logits, dim=-1)
                agv_id = int(torch.argmax(probabilities).item())
                return agv_id
        else:
            raise ValueError(f"Unknown agv_agent_type {self.agv_agent_type}")

    def _get_agv_observation(self):
        current_agv_observation = self.agv_observation_provider.get_observation(
            self.schedule
        )
        agv_obs_tensor = torch.as_tensor(
            current_agv_observation["feat"], dtype=torch.float32
        )
        return agv_obs_tensor

    def _get_next_operation_info(self, action):
        operation_info = self.schedule.get_next_operation_info(action)
        return (
            operation_info.job_id,
            operation_info.op_id,
            operation_info.machine,
            operation_info.duration,
        )

    def get_agent_observation_in_keys(self, agent):
        return self.observation_types[agent + "_0"].get_obervation_in_keys()

    def _get_eligible(self):
        """Get list of eligible jobs (for compatibility with existing code)"""
        return self.schedule.get_eligible_jobs()

    def _get_eligible_agvs(self):
        """Get list of eligible agvs"""
        return self.schedule._get_eligible_agvs()

    def estimate_completion_time(self):
        """Estimate completion time (for compatibility with existing code)"""
        return self.schedule.estimate_completion_time()

    def _calculate_reward(
        self, state_before: Schedule, state_after: Schedule, action_valid: bool = True
    ) -> float:
        """Calculate reward using the configured reward function"""
        reward_data = {
            "state_before": state_before,
            "state_after": state_after,
            "is_complete": self.schedule.is_complete(),
            "step_count": self.step_count,
            "max_episode_steps": self.max_episode_steps,
            "schedule": self.schedule,
            "action_valid": action_valid,
        }
        reward = self.reward_function.calculate_reward(reward_data)
        return {agent: reward for agent in self.possible_agents}

    @functools.cache  # noqa
    def observation_space(self, agent):
        obs_provider = self.observation_types[agent]
        return obs_provider.get_observation_space()

    def observation_space_trl(self, agent):
        obs_provider = self.observation_types[agent]
        return obs_provider.get_observation_space_trl()

    @functools.cache  # noqa
    def action_space(self, agent):
        if self.job_selector_type == JobSelectorType.OPERATION:
            return self.action_space_operation(agent)

        elif self.job_selector_type == JobSelectorType.JOB:
            return self.action_space_job(agent)

        else:
            raise ValueError(
                f"During action_space() Unknown job_selector_type  {self.job_selector_type}"
            )

    def action_space_job(self, agent):
        if self.environment_type == EnvironmentType.SINGLE_AGENT:
            return spaces.Discrete(self.num_jobs)

        else:
            raise ValueError(
                f"During action_space() Unknown environment type {self.environment_type}"
            )

    def action_space_operation(self, agent):
        if self.environment_type == EnvironmentType.SINGLE_AGENT:
            if agent.__contains__("job"):
                return spaces.Discrete(self.num_operations)

        else:
            raise ValueError(
                f"During action_space() Unknown environment type {self.environment_type}"
            )

    def set_schedule_state(self, schedule: Schedule):
        """Set the environment to a specific schedule state"""
        if (
            schedule.num_jobs != self.num_jobs
            or schedule.num_machines != self.num_machines
            or schedule.num_operations != self.num_operations
        ):
            raise ValueError("Schedule dimensions don't match environment")

        self.schedule = schedule.copy()

    def validate_current_schedule(self) -> tuple[bool, list[str]]:
        """Validate the current schedule state"""
        return self.schedule.validate_schedule()

    def set_reward_function(
        self,
        reward_function: str | RewardFunction,
        reward_kwargs: dict | None = None,
    ):
        """
        Change the reward function during runtime

        Args:
            reward_function: Either a string name or RewardFunction instance
            reward_kwargs: Optional kwargs for reward function constructor
        """
        if reward_kwargs is None:
            reward_kwargs = {}

        if isinstance(reward_function, str):
            self.reward_function = get_reward_function(reward_function, **reward_kwargs)
        elif isinstance(reward_function, RewardFunction):
            self.reward_function = reward_function
        else:
            raise ValueError(
                "reward_function must be a string name or RewardFunction instance"
            )

    def get_reward_function_name(self) -> str:
        """Get the name of the current reward function"""
        return self.reward_function.name

    def get_observation_provider_name(self) -> str:
        """Get the name of the current observation provider"""
        return self.observation_provider.name

    def action_mask_single_agent(self):
        mask_jobs = self.action_mask_job_agent()
        mask_agvs = self.action_masks_agvs()
        # return [mask_jobs,mask_agvs]
        return np.concatenate((mask_jobs, mask_agvs))

    def action_mask_job_agent(self):
        if self.job_selector_type == JobSelectorType.JOB:
            return self.schedule.get_valid_job_mask()
        elif self.job_selector_type == JobSelectorType.OPERATION:
            return self.schedule.get_valid_operation_mask()

    def action_masks_agvs(self):
        """
        Generate a mask for valid actions based on eligible jobs for the jobs.
        Returns a binary mask where 1 indicates the job can be scheduled next.
        """
        mask = np.ones(self.num_agvs, dtype=np.int8)

        if sum(mask) == 0:
            raise ValueError("No valid actions available, check the schedule state.")
        return mask

    def set_selected_job_aec(self, action):
        self.schedule.set_selected_job_aec(action)

    def get_observation_type(self, agent):
        return self.observation_types[agent].observation_type
