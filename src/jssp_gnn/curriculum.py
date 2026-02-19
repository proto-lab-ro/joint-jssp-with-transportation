import random

import hydra
from omegaconf import DictConfig

from jssp_core import set_seed
from jssp_core.domain.base_dispatcher import DispatcherBase
from jssp_gnn.dispatcher.utils_matrix import create_environment
from jssp_gnn.utils.setup import create_collector_and_buffer


set_seed(42)


class CurriculumManagerBase:
    def __init__(self, cfg, dispatcher: DispatcherBase = None):
        self.cfg = cfg
        self.dispatcher = dispatcher

    def should_update(self, frames):
        raise NotImplementedError

    def update(self, frames, env, collector, replay_buffer, policy_module):
        raise NotImplementedError

    def _update_jssp_env(self, policy_module, frames):
        # Recreate environment
        new_env = create_environment(self.cfg)

        # Recreate collector and replay buffer
        new_collector, new_replay_buffer = create_collector_and_buffer(
            new_env, policy_module, self.cfg, frames_collected=frames
        )

        return new_env, new_collector, new_replay_buffer

    def _update_agv_env(self, policy_module, frames):
        # Recreate environment

        _ = self.dispatcher.setup_instance()
        new_env = self.dispatcher.setup_environment()
        new_collector = self.dispatcher.setup_collector()
        new_replay_buffer = self.dispatcher.setup_replay_buffer()

        return new_env, new_collector, new_replay_buffer


class None_CurriculumManager(CurriculumManagerBase):
    def __init__(self, cfg, dispatcher: DispatcherBase = None):
        super().__init__(cfg, dispatcher)

    def should_update(self, frames):
        return False

    def update(self, frames, env, collector, replay_buffer, policy_module, device):
        pass


class InstanceBatchCurriculumManager(CurriculumManagerBase):
    """
    Curriculum manager that switches between indexed curriculum elements
    (batches of instances).

    The training loop is expected to call:
        should_update(frames=...) or should_update(gap=...)
        update(frames, env, collector, replay_buffer, policy_module)

    Behavior:
      - should_update(...) returns a boolean:
          True  -> a curriculum switch should happen (self.delta is set accordingly)
          False -> no switch (self.delta is left unchanged; update() will no-op if delta==0)

      - self.delta encodes the next transition:
          * if cfg.curriculum.random_bucket is True:
                self.delta is set so that (current_idx + delta) becomes a randomly chosen
                bucket index in [0, max_idx], excluding the current index.
          * otherwise:
                self.delta = +1 for a forward step (frames-mode always; gap-mode when gap <= forward threshold)
                self.delta = -1 for a backward step (gap-mode when gap >= backward threshold)
                self.delta = 0 means stay.

    Notes:
      - max_idx is inclusive (valid indices are 0..max_idx).
      - update() applies the switch, rebuilds env/collector/replay_buffer, and updates bookkeeping.
    """

    def __init__(
        self,
        cfg,
        element_provider,
        max_idx: int,
        idx_rng: random.Random | None = None,
        dispatcher: DispatcherBase | None = None,
    ):
        super().__init__(cfg, dispatcher)

        self.element_provider = element_provider
        self.max_idx = max_idx
        self.idx_rng = idx_rng or random.Random(42)
        self.delta = 0
        self.terminate: bool = False

        curriculum_cfg = getattr(cfg, "curriculum", None)
        if curriculum_cfg is None:
            raise ValueError(
                "InstanceBatchCurriculumManager requires cfg.curriculum to be set."
            )

        self.initial_idx = getattr(curriculum_cfg, "initial_idx", 0)
        self.used_bucket_idx = {self.initial_idx}
        self.element = self.element_provider(self.cfg, self.initial_idx)
        print(
            f"\nInitial curriculum index: {self.initial_idx} , Starting with size: {self.element.sample_size}\n"
        )

        self.current_idx = self.initial_idx
        self.mode = getattr(curriculum_cfg, "mode", "frames")

        # Frames-based mode
        self.stage_frames = getattr(curriculum_cfg, "stage_frames", None)

        # Gap-based mode
        self.forward_gap_threshold = getattr(
            curriculum_cfg, "forward_gap_threshold", None
        )
        self.backward_gap_threshold = getattr(
            curriculum_cfg, "backward_gap_threshold", None
        )

        self.last_switch_frame: int = 0

        self.flexible_rollout = getattr(curriculum_cfg, "flexible_rollout", False)
        self.flexible_rollout_max_ops = getattr(
            curriculum_cfg, "flexible_rollout_max_ops", 100
        )

    def should_update(
        self, *args, frames: int | None = None, gap: float | None = None
    ) -> bool:
        """
        Decide whether to switch curriculum element.

        Returns:
            True  -> switch should happen; self.delta is set to the step size.
            False -> no switch; update() will no-op if self.delta == 0.
        """
        # Backward compatibility: positional frames
        if args:
            if len(args) != 1:
                raise TypeError(
                    "should_update() accepts at most one positional argument (frames)."
                )
            frames = args[0]

        if self.mode == "frames":
            return self._should_update_frames(frames)
        elif self.mode == "gap":
            return self._should_update_gap(gap)
        else:
            self.delta = 0
            return False

    def _should_update_frames(self, frames: int | None) -> bool:
        if frames is None:
            print("- Frames is None!!\n\n")
            self.delta = 0
            return False
        if self.stage_frames is None:
            print("- Stage frames not set!!\n\n")
            self.delta = 0
            return False

        if frames - self.last_switch_frame < self.stage_frames:
            # print("- Not enough frames elapsed!!\n\n")
            self.delta = 0
            return False

        proposed_idx = self.current_idx + 1
        if self.max_idx is not None and proposed_idx > self.max_idx:
            print(
                f"- Max index Reached!!, Forward curriculum step not possible - staying on current!! | Current Stage : {self.current_idx + 1} | Current Size: {self.element.sample_size}\n\n"
            )
            self.terminate = True
            self.delta = 0
            return False

        if self.cfg.curriculum.random_bucket:
            print(
                "- Random bucket curriculum step identified!!, moving to random index..."
            )
            random_idx = self.idx_rng.choice(
                [i for i in range(0, self.max_idx + 1) if i not in self.used_bucket_idx]
            )
            self.delta = (
                random_idx - self.current_idx
            )  # move to the random proposed index first
            return True

        print("- Forward curriculum step identified!!, moving forward...")
        self.delta = +1
        return True

    def _should_update_gap(self, gap: float | None) -> bool:
        print(
            "\n\n=============  CHECKING GAP VALUES IN CURRICULUM MANAGER  ============="
        )
        if gap is None:
            print("- Gap is None!!")
            self.delta = 0
            return False
        if self.forward_gap_threshold is None or self.backward_gap_threshold is None:
            print("- Gap thresholds not set!!")
            self.delta = 0
            return False

        print(
            f"- Current gap: {gap:.3f} | Current threshold ({self.forward_gap_threshold}, {self.backward_gap_threshold})"
        )
        if gap <= self.forward_gap_threshold:
            if self.cfg.curriculum.random_bucket:
                print(
                    "- Random bucket curriculum step identified!!, moving to random index..."
                )
                random_idx = self.idx_rng.choice(
                    [
                        i
                        for i in range(0, self.max_idx + 1)
                        if i not in self.used_bucket_idx
                    ]
                )
                self.delta = (
                    random_idx - self.current_idx
                )  # move to the random proposed index first
                return True

            self.delta = +1
            proposed_idx = self.current_idx + self.delta

            # check max idx:
            if self.max_idx is not None and proposed_idx > self.max_idx:
                print(
                    f"- Max index Reached!!, Forward curriculum step not possible - staying on current!! | Current Stage: {self.current_idx + 1} | Current Size: {self.element.sample_size}"
                )
                print(
                    "-----------------------------------------------------------------------\n\n"
                )
                self.delta = 0
                self.terminate = True
                return False
            print("- Forward curriculum step identified!!, moving forward...")
            return True
        elif gap >= self.backward_gap_threshold:
            if self.cfg.curriculum.random_bucket:
                print(
                    "- Random bucket curriculum step identified!!, moving to random index..."
                )
                random_idx = self.idx_rng.choice(
                    [
                        i
                        for i in range(0, self.max_idx + 1)
                        if i not in self.used_bucket_idx
                    ]
                )
                self.delta = (
                    random_idx - self.current_idx
                )  # move to the random proposed index first
                return True

            self.delta = -1
            proposed_idx = self.current_idx + self.delta
            # check min idx:
            if proposed_idx < 0:
                print(
                    f"- Min index Reached!!, Backward curriculum step not possible - staying on current!! | Current Stage: {self.current_idx + 1} | Current Size: {self.element.sample_size}"
                )
                print(
                    "-----------------------------------------------------------------------\n\n"
                )
                self.delta = 0
                return False
            print("- Backward curriculum step identified!!, moving backward...")
            return True
        else:
            print(
                f"- Gap within thresholds!!, staying on current. | Current Stage: {self.current_idx + 1} | Current Size: {self.element.sample_size}\n\n"
            )
            self.delta = 0
            return False

        # return True

    def update(self, frames: int, env, collector, replay_buffer, policy_module):
        """
        Apply a curriculum transition decided by should_update().

        Args: delta: -1 (previous), 0 (stay), +1 (next)
        frames: current rollout_frames (for bookkeeping)
        env, collector, replay_buffer: current RL components
        policy_module, device: used to rebuild collector/buffer

        Returns: new_env, new_collector, new_replay_buffer
        """

        if self.delta == 0:
            return env, collector, replay_buffer

        target_idx = self.current_idx + self.delta

        if self.max_idx is not None:
            if target_idx < 0 or target_idx > self.max_idx:
                return env, collector, replay_buffer

        try:
            self.element = self.element_provider(self.cfg, target_idx)
        except Exception as e:
            print(f"[Curriculum] Failed to obtain element for index {target_idx}: {e}")
            print(
                "-----------------------------------------------------------------------\n\n"
            )
            return env, collector, replay_buffer

        print(
            f"\n[Curriculum] Switching from element {self.current_idx} "
            f"to {target_idx} using element: {self.element.sample_size}, from file: {self.element.instances_file}"
        )

        print(
            "--------------------------------------------------------------------- \n\n"
        )

        self.current_idx = target_idx
        self.used_bucket_idx.add(self.current_idx)
        self.last_switch_frame = frames

        # Apply curriculum element to config
        self.cfg.env.instance = self.element.sample_size

        if self.flexible_rollout:
            if self.element.num_ops > self.flexible_rollout_max_ops:
                self.cfg.training.frames_per_batch = self.element.num_ops * 5
                self.cfg.training.sub_batch_size = self.element.num_ops * 5
        # when n_epoch=1
        # self.cfg.training.eval_freq = self.cfg.training.frames_per_batch * 6
        # self.cfg.training.eval_freq = eval_freq_by_sample_size_2028.get(
        #     self.element.sample_size, 6144
        # )
        self.cfg.env.instance_generator_kwargs = {
            "instances_file": str(self.element.instances_file)
        }

        # Cleanup old components
        if env is not None:
            env.close()
            del env
        if collector is not None:
            collector.shutdown()
            del collector
        if replay_buffer is not None:
            del replay_buffer

        # Rebuild environment stack
        if self.dispatcher is None:
            new_env, new_collector, new_replay_buffer = self._update_jssp_env(
                policy_module, frames
            )
        else:
            self.dispatcher.cfg = self.cfg
            new_env, new_collector, new_replay_buffer = self._update_agv_env(
                policy_module, frames
            )

        return new_env, new_collector, new_replay_buffer


def get_curriculum_manager(cfg, dispatcher: DispatcherBase = None):
    """
    Get the appropriate curriculum manager based on config.

    Returns None_CurriculumManager if no curriculum config exists,
    Interval_CurriculumManager if interval config is present,
    otherwise Stages_CurriculumManager.
    """
    # Check if curriculum attribute exists in config
    if not hasattr(cfg, "curriculum") or cfg.curriculum is None:
        return None_CurriculumManager(cfg, dispatcher)

    # Default to stages curriculum manager
    return None_CurriculumManager(cfg, dispatcher)


if __name__ == "__main__":

    @hydra.main(
        version_base=None, config_path="../../conf/gnn", config_name="jh_train"
    )  # mo_train
    def test(cfg: DictConfig):
        dispatcher = None
        cm = get_curriculum_manager(cfg, dispatcher)
        print(f"Using curriculum manager: {type(cm).__name__}")

    # Configuration for the batch experiments
    test()
