from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

from jssp_core.domain import (
    EnvironmentType,
    JobSelectorType,
)
from jssp_core.environments.agv_parallel import JsspAGVEnv
from jssp_core.instances import (
    InstanceGeneratorLike,
)
from jssp_core.observation_providers import ObservationProvider
from jssp_core.reward_functions import RewardFunction


class JsspAGVEnvAEC(AECEnv):
    metadata = {"name": "jsspagvenvaec"}

    def __init__(
        self,
        instance,
        max_episode_steps=2000,
        random_instance=False,
        reward_function: str | RewardFunction = "sparse_exponential",
        reward_kwargs: dict | None = None,
        job_observation_provider: str | ObservationProvider = "default",
        job_observation_kwargs: dict | None = None,
        agv_observation_provider: str | ObservationProvider = "default",
        agv_observation_kwargs: dict | None = None,
        environment_type: EnvironmentType = EnvironmentType.MULTI_AGENT,
        number_agvs=3,
        render_mode=None,
        job_selector_type: JobSelectorType = JobSelectorType.JOB,
        instance_generator: InstanceGeneratorLike | None = None,
        instance_generator_kwargs: dict | None = None,
    ):
        self.parallel_env = JsspAGVEnv(
            instance,
            max_episode_steps=max_episode_steps,
            random_instance=random_instance,
            reward_function=reward_function,
            reward_kwargs=reward_kwargs,
            job_observation_provider=job_observation_provider,
            job_observation_kwargs=job_observation_kwargs,
            agv_observation_provider=agv_observation_provider,
            agv_observation_kwargs=agv_observation_kwargs,
            environment_type=environment_type,
            number_agvs=number_agvs,
            render_mode=render_mode,
            job_selector_type=job_selector_type,
            instance_generator=instance_generator,
            instance_generator_kwargs=instance_generator_kwargs,
        )
        self.job_selector_type = job_selector_type

        self.NONE = 99
        self._agent_selector = AgentSelector(self.agents)
        self._cumulative_rewards = None

    @property
    def random_instance(self):
        return self.parallel_env.random_instance

    @property
    def possible_agents(self):
        return self.parallel_env.possible_agents

    @property
    def agents(self):
        return self.parallel_env.possible_agents[:]

    @property
    def group_map(self):
        return self.parallel_env.group_map

    @property
    def num_operations(self):
        return self.parallel_env.num_operations

    @property
    def num_jobs(self):
        return self.parallel_env.num_jobs

    @property
    def original_instance(self):
        return self.parallel_env.original_instance

    @property
    def agent_name_mapping(self):
        return self.parallel_env.agent_name_mapping

    @property
    def schedule(self):
        return self.parallel_env.schedule

    @property
    def num_agvs(self):
        return self.parallel_env.num_agvs

    @property
    def num_machines(self):
        return self.parallel_env.num_machines

    @num_agvs.setter
    def num_agvs(self, value):
        self.schedule.num_agvs = value

    @original_instance.setter
    def original_instance(self, value):
        self.parallel_env.original_instance = value

    @property
    def instance(self):
        return self.parallel_env.instance

    @instance.setter
    def instance(self, new_instance):
        self.parallel_env.instance = new_instance

    @property
    def observation_provider(self):
        return self.parallel_env.observation_provider

    @property
    def job_observation_kwargs(self):
        return self.parallel_env.job_observation_kwargs

    @property
    def agv_observation_provider(self):
        return self.parallel_env.agv_observation_provider

    @property
    def agv_observation_kwargs(self):
        return self.parallel_env.agv_observation_kwargs

    def state_summary(self):
        print("\n", "===== Current State Summary =====")
        print(f"self.possible_agents {self.possible_agents}")
        print(f"self.agent_name_mapping  {self.agent_name_mapping}")
        print(f"self.agents  {self.agents}")
        print(f"self.agent_selection  {self.agent_selection}")
        print(f"self.observations  {self.observations}")
        print(f"self.truncations  {self.truncations}")
        print(f"self.terminations  {self.terminations}")
        print(f"self.rewards  {self.rewards}")
        print(f"self.state  {self.state}")
        print(f"self._cumulative_rewards  {self._cumulative_rewards}")

    # Reset and Step do not return observations or other dicts -> They fill dicts which are then called!
    def reset(self, seed=None, options=None):
        self.observations, self.infos = self.parallel_env.reset()
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.num_moves = 0
        self.state = {agent: self.NONE for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        # stores action of current agent

        self.state[agent] = action
        if self._agent_selector.is_last():
            self._set_selected_job_aec(None)
            (
                self.observations,
                self.rewards,
                self.terminations,
                self.truncations,
                self.infos,
            ) = self.parallel_env.step(
                {self.agents[0]: self.state[self.agents[0]], self.agents[1]: action},
            )

            self.num_moves += 1

        else:
            # necessary so that observe() returns a reasonable observation at all times.
            other_agent_name = self.agents[1 - self.agent_name_mapping[agent]]
            self.state[other_agent_name] = self.NONE
            if self.job_selector_type == JobSelectorType.OPERATION:
                if self.agent_selection.__contains__("job"):
                    action = self.schedule.flat_index_to_job_op(action)[0]
            self._set_selected_job_aec(action)
            self.observations[other_agent_name] = self._get_obs()[other_agent_name]

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def _set_selected_job_aec(self, action):
        self.parallel_env.set_selected_job_aec(action)

    def state(self):
        raise NotImplementedError("state not implemented yet.")

    def _get_obs(self):
        return self.parallel_env._get_obs()

    def observe(self, agent):
        return self.observations[agent]

    def observation_space(self, agent):
        return self.parallel_env.observation_space(agent)

    def action_space(self, agent):
        return self.parallel_env.action_space(agent)

    def observation_space_trl(self, agent):
        return self.parallel_env.observation_space_trl(agent)

    def get_observation_type(self, agent):
        return self.parallel_env.observation_types[agent].observation_type

    def get_agent_observation_in_keys(self, agent):
        return self.parallel_env.get_agent_observation_in_keys(agent)


if __name__ == "__main__":
    # Create a random instance and an environment, then run one episode with random actions.
    from jssp_core.instances import (
        InstanceGeneratorLike,
        generate_random_transport_instance,
    )

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
        "job_observation_provider": "lb_bipartite_gnn",
        "job_observation_kwargs": {
            "self_loop": False,
            "observation_type": "graph_matrix",
            "normalize": "operation",
        },
        "agv_observation_provider": "default",
        "agv_observation_kwargs": {"observation_type": "dict"},
    }

    env = JsspAGVEnvAEC(instance, number_agvs=3, **config)

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
        env.reset()

        print(f"---env.num_agvs: {env.num_agvs}")
        # print(
        #     f'---obs["agv_selector_0"]["feat"].shape[0]: {obs["agv_selector_0"]["feat"].shape[0]}'
        # )
        print(
            f"---env.num_jobs, num_operations,num_machines : {env.num_jobs}, {env.num_operations} {env.num_machines}"
        )
