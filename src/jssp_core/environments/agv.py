import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv
from torch import nn

from jssp_core.domain import (
    EnvironmentType,
    ItemDataType,
    JobAgentType,
    JobSelectorType,
)
from jssp_core.domain.domains import ObservationType
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


class AgvEnv(ParallelEnv):
    """
    Gymnasium environment for Job Shop Scheduling Problem (JSSP).
    Uses the Schedule class for core scheduling logic.
    """

    def __init__(
        self,
        instance,
        max_episode_steps=2000,
        random_instance=True,
        reward_function: str | RewardFunction = "sparse_exponential",
        reward_kwargs: dict | None = None,
        job_selector: nn.Module | Heuristic = None,
        job_selector_kwargs: dict | None = None,
        agv_observation_provider: str | ObservationProvider = "default",
        agv_observation_kwargs: dict | None = None,
        number_agvs=3,
        render_mode=None,
        instance_generator: InstanceGeneratorLike | None = None,
        instance_generator_kwargs: dict | None = None,
    ):
        self.environment_type = EnvironmentType.SINGLE_AGENT
        self.job_selector_kwargs = job_selector_kwargs or {}
        self.job_selector = job_selector
        self._instance_generator: InstanceGenerator | None = None
        self._instance_generator_spec = instance_generator
        self._instance_generator_kwargs = dict(instance_generator_kwargs or {})
        self.possible_agents = ["agv_selector_0"]
        self.agents = self.possible_agents.copy()
        # optional: a mapping between agent name and ID
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
        self.agv_observation_kwargs = agv_observation_kwargs or {}

        # Initialize with the provided instance to get dimensions
        self.schedule = TransportSchedule(instance, num_agvs=number_agvs)

        self.action_selected_job_id = 99

        if self.random_instance:
            self._instance_generator = ensure_instance_generator(
                self._instance_generator_spec,
                num_jobs=self.num_jobs,
                num_machines=self.num_machines - 2,
                generator_kwargs=self._instance_generator_kwargs,
            )
        self.set_up_job_selector()
        self.set_up_reward_function(reward_kwargs)

        self.set_up_observation_provider(agv_observation_provider)

        # Aber die Abfrage des Typs soll raus
        self.observation_types = {}
        for agent in self.possible_agents:
            if agent.__contains__("job"):
                raise NotImplementedError(
                    "Job observation provider not implemented yet."
                )
            elif agent.__contains__("agv"):
                self.observation_types[agent] = self.agv_observation_provider

        # Gym-specific attributes
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

    def set_up_job_selector(self):
        if isinstance(self.job_selector, Heuristic):
            self.job_agent_type = JobAgentType.HEURISTIC
        elif isinstance(self.job_selector, nn.Module):
            self.job_agent_type = JobAgentType.POLICY

        else:
            raise ValueError("job_selector must be Heuristic or nn.Module instance")
        self.set_up_job_observation_provider()

    def set_up_job_observation_provider(self):
        job_observation_provider = self.job_selector_kwargs["job_observation_provider"]
        job_observation_kwargs = self.job_selector_kwargs["job_observation_kwargs"]
        self.job_policy_observation_type = job_observation_kwargs["observation_type"]
        if isinstance(job_observation_provider, str):
            self.job_observation_provider = get_observation_provider(
                job_observation_provider, self.schedule, **job_observation_kwargs
            )
        elif isinstance(job_observation_provider, ObservationProvider):
            self.job_observation_provider = job_observation_provider
        else:
            raise ValueError(
                "job_observation_provider must be a string name or ObservationProvider instance"
            )

    def set_up_reward_function(self, reward_kwargs):
        # Set up reward function
        if reward_kwargs is None:
            reward_kwargs = {}
        self.truncate_if_invalid = reward_kwargs.get("truncate_if_invalid", False)

        if isinstance(self.reward_function, str):
            # Pass the initial TransportSchedule to the reward function constructor
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

    def set_up_observation_provider(self, agv_observation_provider):
        if isinstance(agv_observation_provider, str):
            self.agv_observation_provider = get_agv_observation_provider(
                agv_observation_provider, self.schedule, **self.agv_observation_kwargs
            )
        elif isinstance(agv_observation_provider, ObservationProvider):
            self.agv_observation_provider = agv_observation_provider
        else:
            raise ValueError(
                "agv_observation_provider must be a string name or ObservationProvider instance"
            )

    def _get_info(self):
        """Get additional info about the environment state"""

        current_makespan = self.get_makespan()
        info_dict = {}
        # if self.environment_type == EnvironmentType.SINGLE_AGENT:
        action_mask_single_agent = self.action_masks_agvs()
        info_dict[self.possible_agents[0]] = {}
        info_dict[self.possible_agents[0]]["action_mask"] = action_mask_single_agent
        info_dict[self.possible_agents[0]]["makespan"] = current_makespan

        jssp_obs = self._get_job_observation()
        if self.job_policy_observation_type == ObservationType.GRAPH_MATRIX:
            node_feats = jssp_obs[0]
            edge_index = jssp_obs[1]
            info_dict[self.possible_agents[0]]["critic_jssp_node_feats"] = node_feats
            info_dict[self.possible_agents[0]]["critic_jssp_edge_index"] = edge_index
        else:
            info_dict[self.possible_agents[0]]["critic_jssp"] = jssp_obs

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
            raise RuntimeError("Selected job cannot be scheduled: {action_job}")

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
        self.action_selected_job_id = 99

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
        self.agv_observation_provider.reset(self.schedule)
        self.job_observation_provider.reset(self.schedule)
        self.step_count = 0
        self._get_job_action()

        return self._get_obs(), self._get_info()

    def _get_job_observation(self):
        current_job_observation = self.job_observation_provider.get_observation(
            self.schedule
        )

        if self.job_policy_observation_type == ObservationType.FLAT:
            if isinstance(current_job_observation, np.ndarray):
                obs_tensor = torch.from_numpy(current_job_observation).float()
        elif self.job_policy_observation_type == ObservationType.DICT:
            raise NotImplementedError("DICT observation type not implemented yet.")
        elif self.job_policy_observation_type == ObservationType.GRAPH:
            obs_tensor = Data(
                x=torch.from_numpy(current_job_observation["node_feats"]).float(),
                edge_index=torch.from_numpy(
                    current_job_observation["edge_index"]
                ).long(),
            )
            raise UserWarning("GRAPH observation type not tested yet.")

        elif self.job_policy_observation_type == ObservationType.GRAPH_MATRIX:
            # node_feats, edge_index
            obs_tensor = (
                torch.from_numpy(current_job_observation["node_feats"]).float(),
                torch.from_numpy(current_job_observation["edge_index"]).long(),
            )

        return obs_tensor

    def _get_job_action(self):
        """Select the next job to be scheduled based on the job selector type"""
        if self.job_agent_type == JobAgentType.HEURISTIC:
            if self.job_selector.item_type == ItemDataType.JOB:
                selected_job_id = self.job_selector.step(self.schedule)

            elif self.job_selector.item_type == ItemDataType.OP:
                raise NotImplementedError(
                    "Operation-based job selector not implemented yet."
                )

        elif self.job_agent_type == JobAgentType.POLICY:
            if self.job_selector_kwargs["job_selector_type"] == ItemDataType.OP:
                action_mask = self.schedule.get_valid_operation_mask()
            elif self.job_selector_kwargs["job_selector_type"] == ItemDataType.JOB:
                action_mask = self.schedule.get_valid_job_mask()
            # raise NotImplementedError("Policy-based job selector not implemented yet.")
            obs_tensor = self._get_job_observation()
            current_job_observation = obs_tensor
            with torch.no_grad():
                self.job_selector.eval()
                if self.job_policy_observation_type == ObservationType.GRAPH_MATRIX:
                    logits = self.job_selector(
                        current_job_observation[0],
                        current_job_observation[1],
                        action_mask,
                    )
                else:
                    logits = self.job_selector(current_job_observation)

            # Apply action mask
            masked_logits = logits.masked_fill(
                ~torch.tensor(action_mask, dtype=torch.bool), float("-inf")
            )

            # Validate shape
            expected_shape = (1, self.schedule.num_operations)
            assert (
                masked_logits.shape == expected_shape
                or masked_logits.shape == logits.shape
            ), (
                f"masked_logits shape mismatch: expected {expected_shape}, "
                f"got {masked_logits.shape}"
            )
            probabilities = torch.softmax(masked_logits, dim=-1)
            selected_job_id = int(torch.argmax(probabilities).item())

            if self.job_selector_kwargs["job_selector_type"] == ItemDataType.JOB:
                selected_job_id = selected_job_id
            elif self.job_selector_kwargs["job_selector_type"] == ItemDataType.OP:
                selected_job_id = self.schedule.flat_index_to_job_op(selected_job_id)[0]
        else:
            raise ValueError(f"Unknown job_agent_type {self.job_agent_type}")

        self.action_selected_job_id = selected_job_id
        self.set_selected_job_aec(selected_job_id)

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
        # print(f' Step Count: {self.step_count}')

        action_agv = actions["agv_selector_0"]
        action_job = self.action_selected_job_id

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
        self._validate_agv_action(action_agv)

        state_before = self.schedule.copy()

        # Schedule the operation
        success = self.schedule.schedule_job(action_job, **{"agv": action_agv})

        if not success:
            raise ValueError(
                f"Tried to schedule non-permitted operation {action_job}, with AGV: {action_agv}, "
            )

        state_after = self.schedule

        # Calculate reward
        reward_dict = self._calculate_reward(state_before, state_after)

        # Check if episode is done
        terminated = self.schedule.is_complete()
        terminated_dict = {agent: terminated for agent in self.possible_agents}
        truncated_dict = {agent: False for agent in self.possible_agents}
        if not terminated:
            self._get_job_action()

        return (
            self._get_obs(),
            reward_dict,
            terminated_dict,
            truncated_dict,
            self._get_info(),
        )

    def _get_next_operation_info(self, action):
        operation_info = self.schedule.get_next_operation_info(action)
        return (
            operation_info.job_id,
            operation_info.op_id,
            operation_info.machine,
            operation_info.duration,
        )

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

    def observation_space(self, agent):
        if not hasattr(self, "_observation_space_cache"):
            self._observation_space_cache = {}
        if agent not in self._observation_space_cache:
            obs_provider = self.observation_types[agent]
            self._observation_space_cache[agent] = obs_provider.get_observation_space()
        return self._observation_space_cache[agent]

    def observation_space_trl(self, agent):
        obs_provider = self.observation_types[agent]
        return obs_provider.get_observation_space_trl()

    def action_space(self, agent):
        if not hasattr(self, "_action_space_cache"):
            self._action_space_cache = {}
        if agent not in self._action_space_cache:
            self._action_space_cache[agent] = spaces.Discrete(self.num_agvs)
        return self._action_space_cache[agent]

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

    def action_masks_agvs(self):
        """
        Generate a mask for valid actions based on eligible jobs for the jobs.
        Returns a binary mask where 1 indicates the job can be scheduled next.
        """
        mask = np.ones(self.num_agvs, dtype=np.int8)

        if sum(mask) == 0:
            raise ValueError("No valid actions available, check the schedule state.")
        return mask

    def action_mask_job_agent(self):
        if self.job_selector_kwargs["job_selector_type"] == JobSelectorType.JOB:
            return self.schedule.get_valid_job_mask()
        elif self.job_selector_kwargs["job_selector_type"] == JobSelectorType.OPERATION:
            return self.schedule.get_valid_operation_mask()

    def set_selected_job_aec(self, action):
        self.schedule.set_selected_job_aec(action)

    def get_observation_type(self, agent):
        return self.observation_types[agent].observation_type

    def get_agent_observation_in_keys(self, agent):
        return self.observation_types[agent + "_0"].get_obervation_in_keys()


if __name__ == "__main__":
    from jssp_core.domain.domains import ObservationType

    # # Create a random instance and an environment, then run one episode with random actions.
    # instance = _load_transport_instance()
    # env = AgvEnv(
    #     instance,
    #     random_instance=False,
    #     number_agvs=3,
    #     max_episode_steps=200,
    #     job_selector=job_heuristic_factory("SPT"),
    #     agv_observation_provider="agv_aec_selM_allM",
    #     agv_observation_kwargs={"observation_type": ObservationType.DICT},
    # )
    # obs, info = env.reset()
    # done = False
    # i = 0
    # while not done:
    #     actions = {}
    #     mask = info["agv_selector_0"]["action_mask"]
    #     for agent in env.agents:
    #         # select a valid action index from the action mask
    #         mask = info[agent]["action_mask"]
    #         valid = np.flatnonzero(mask)
    #         if valid.size == 0:
    #             actions[agent] = env.action_space(agent).sample()
    #         else:
    #             actions[agent] = int(np.random.choice(valid))
    #     obs, rewards, terminated, truncated, info = env.step(actions)
    #     done = all(terminated.values()) or all(truncated.values())
    #     i += 1
    # print("Episode finished.")
    # print("Final makespan:", env.get_makespan())
    # Create a random instance and an environment, then run one episode with random actions.
    from jssp_core.instances import (
        generate_random_transport_instance,
    )
    from jssp_core.solver.heuristics import job_heuristic_factory

    instance = generate_random_transport_instance(
        num_jobs=6,
        num_machines=6,
        min_duration=1,
        max_duration=10,
    )
    nr_agvs = [1, 3]
    config = {
        # "instance": "jssp_instances/transport/ft06",
        "random_instance": False,
        "instance_generator": "transport_random_uniform",
        "instance_generator_kwargs": {
            "min_duration": 1,
            "max_duration": 100,
        },
        "reward_function": "sparse_makespan",
        "reward_kwargs": {},
        "job_selector": job_heuristic_factory("SPT"),
        "job_selector_kwargs": {
            "job_observation_provider": "default",
            "job_observation_kwargs": {"observation_type": "flat"},
        },
        "agv_observation_provider": "default",
        "agv_observation_kwargs": {"observation_type": "dict"},
    }

    env = AgvEnv(instance, number_agvs=3, **config)

    for nr_agv in nr_agvs:
        instance = generate_random_transport_instance(
            num_jobs=nr_agv,
            num_machines=nr_agv,
            min_duration=1,
            max_duration=10,
        )

        print(f" Running episode with {nr_agv} AGVs ---")
        env.num_agvs = nr_agv
        env.instance = instance
        print(
            f"---env.num_jobs, num_operations,num_machines : {env.num_jobs}, {env.num_operations} {env.num_machines}"
        )
        print("--- Resetting environment ---")
        obs, _ = env.reset()

        print(f"---env.num_agvs: {env.num_agvs}")
        # print(
        #     f'---obs["agv_selector_0"]["feat"].shape[0]: {obs["agv_selector_0"]["feat"].shape[0]}'
        # )
        print(
            f"---env.num_jobs, num_operations,num_machines : {env.num_jobs}, {env.num_operations} {env.num_machines}"
        )
