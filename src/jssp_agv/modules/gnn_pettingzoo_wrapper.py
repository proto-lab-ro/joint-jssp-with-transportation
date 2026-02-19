import torch
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform
from torchrl.envs.libs.pettingzoo import PettingZooWrapper


class AgvGnnPettingZooWrapper(PettingZooWrapper):
    def __init__(
        self,
        env,
        return_state: bool = False,
        group_map=None,
        use_mask: bool = False,
        categorical_actions: bool = True,
        seed: int | None = None,
        done_on_any: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            env=env,
            return_state=return_state,
            group_map=group_map,
            use_mask=use_mask,
            categorical_actions=categorical_actions,
            seed=seed,
            done_on_any=done_on_any,
            **kwargs,
        )

    def _make_group_specs(self, group_name: str, agent_names: list[str]):
        n_agents = len(agent_names)
        action_specs = []
        observation_specs = []
        for agent in agent_names:
            action_specs.append(
                Composite(
                    {
                        "action": _gym_to_torchrl_spec_transform(
                            self.action_space(agent),
                            remap_state_to_observation=False,
                            categorical_action_encoding=self.categorical_actions,
                            device=self.device,
                        )
                    },
                )
            )
            observation_specs.append(
                Composite(
                    {  # self._env.observation_provider.get_observation_space_trl()
                        "observation": self._env.observation_space_trl(agent)
                    }
                )
            )
        group_action_spec = torch.stack(action_specs, dim=0)
        group_observation_spec = torch.stack(observation_specs, dim=0)

        # Sometimes the observation spec contains an action mask.
        # Or sometimes the info spec contains an action mask.
        # We uniform this by removing it from both places and optionally set it in a standard location.
        group_observation_inner_spec = group_observation_spec["observation"]
        if (
            isinstance(group_observation_inner_spec, Composite)
            and "action_mask" in group_observation_inner_spec.keys()
        ):
            self.has_action_mask[group_name] = True
            del group_observation_inner_spec["action_mask"]
            group_observation_spec["action_mask"] = Categorical(
                n=2,
                shape=(
                    group_action_spec["action"].shape
                    if not self.categorical_actions
                    else group_action_spec["action"].to_one_hot_spec().shape
                ),
                dtype=torch.bool,
                device=self.device,
            )

        if self.use_mask:
            group_observation_spec["mask"] = Categorical(
                n=2,
                shape=torch.Size((n_agents,)),
                dtype=torch.bool,
                device=self.device,
            )

        group_reward_spec = Composite(
            {
                "reward": Unbounded(
                    shape=torch.Size((n_agents, 1)),
                    device=self.device,
                    dtype=torch.float32,
                )
            },
            shape=torch.Size((n_agents,)),
        )
        group_done_spec = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            shape=torch.Size((n_agents,)),
        )
        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
            group_done_spec,
        )
