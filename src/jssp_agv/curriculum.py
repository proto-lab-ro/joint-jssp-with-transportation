import random

from omegaconf import OmegaConf

from jssp_core import set_seed
from jssp_core.domain.base_dispatcher import DispatcherBase
from jssp_gnn.curriculum import CurriculumManagerBase, get_curriculum_manager


set_seed(42)


class Stages_CurriculumManager(CurriculumManagerBase):
    def __init__(self, cfg, dispatcher: DispatcherBase = None):
        super().__init__(cfg, dispatcher)

        self.stages = sorted(
            cfg.get("curriculum", []), key=lambda x: x.get("start_frame", 0)
        )
        self.current_stage_idx = 0

    def should_update(self, frames):
        if self.current_stage_idx < len(self.stages):
            stage = self.stages[self.current_stage_idx]
            if frames >= stage.get("start_frame", 0):
                return True
        return False

    def update(self, frames, env, collector, replay_buffer, policy_module):
        stage = self.stages[self.current_stage_idx]
        print(f"Switching to curriculum stage {self.current_stage_idx}: {stage}")

        # Update config
        if "env" in stage:
            if self.dispatcher is not None:
                if stage.env.number_agvs == "random":
                    stage.env.number_agvs = _get_random_number_agvs()
            print(f" ------> Stage env {stage.env}")
            self.cfg.env = OmegaConf.merge(self.cfg.env, OmegaConf.create(stage.env))

        # Clean up old components
        if env is not None:
            env.close()  # Close old env if needed
            del env
        if collector is not None:
            collector.shutdown()
            del collector
        if replay_buffer is not None:
            del replay_buffer
        if self.dispatcher is None:
            new_env, new_collector, new_replay_buffer = self._update_jssp_env(
                policy_module, frames
            )
        elif self.dispatcher is not None:
            self.dispatcher.cfg = self.cfg
            new_env, new_collector, new_replay_buffer = self._update_agv_env(
                policy_module, frames
            )
        else:
            raise ValueError("Dispatcher must be provided for AGV curriculum updates.")

        self.current_stage_idx += 1
        return new_env, new_collector, new_replay_buffer


class Interval_CurriculumManager(CurriculumManagerBase):
    def __init__(self, cfg, dispatcher: DispatcherBase = None):
        super().__init__(cfg, dispatcher)
        self.update_interval = self.cfg.curriculum.interval.get(
            "update_interval_frames", None
        )

        self.last_update_frame = 0
        self.number_agvs_type = self.cfg.curriculum.env.get("number_agvs", None)
        self.instance_list = self.cfg.curriculum.env.get("instance", [])

    def should_update(self, frames):
        """Check if curriculum should update based on frame interval."""
        if self.update_interval is None:
            return False

        return frames - self.last_update_frame >= self.update_interval

    def update(self, frames, env, collector, replay_buffer, policy_module):
        """Update curriculum and track the update frame."""
        stage = {}
        if self.number_agvs_type == "random":
            stage["number_agvs"] = _get_random_number_agvs()

        if self.instance_list:
            stage["instance"] = _select_random_instance(self.instance_list)

        print(f" ------> Stage env :{stage}")
        self.cfg.env = OmegaConf.merge(self.cfg.env, OmegaConf.create(stage))

        if env is not None:
            env.close()
            del env
        if collector is not None:
            collector.shutdown()
            del collector
        if replay_buffer is not None:
            del replay_buffer
        if self.dispatcher is None:
            new_env, new_collector, new_replay_buffer = self._update_jssp_env(
                policy_module, frames
            )
        elif self.dispatcher is not None:
            self.dispatcher.cfg = self.cfg
            new_env, new_collector, new_replay_buffer = self._update_agv_env(
                policy_module, frames
            )
        else:
            raise ValueError("Dispatcher must be provided for AGV curriculum updates.")

        self.last_update_frame = frames

        self._set_update_interval(new_env)

        return new_env, new_collector, new_replay_buffer

    def _set_update_interval(self, env):
        """Set the update interval based on environment instance size."""
        pass


class NrInstances_CurriculumManager(CurriculumManagerBase):
    def __init__(self, cfg, dispatcher: DispatcherBase = None):
        super().__init__(cfg, dispatcher)

        self.last_update_frame = 0
        self.number_agvs_type = self.cfg.curriculum.env.get("number_agvs", None)
        self.instance_list = self.cfg.curriculum.env.get("instance", [])
        self.nr_instances_multiplier = (
            self.cfg.curriculum.interval.nr_instances_multiplier
        )
        self.sub_batch_size_1 = self.cfg.curriculum.interval.sub_batch_size_1
        self.nr_repeat_rollout = self.cfg.curriculum.interval.nr_repeat_rollout
        self.update_interval = (
            self._get_frames_per_batch("jssp_instances/transport/ft06")
            * self.nr_repeat_rollout
        )

    def should_update(self, frames):
        """Check if curriculum should update based on frame interval."""
        if self.update_interval is None:
            return False
        return frames - self.last_update_frame >= self.update_interval

    def _get_frames_per_batch(self, instance):
        total_ops = _instance_num_operations(instance)
        return total_ops * self.nr_instances_multiplier

    def update(self, frames, env, collector, replay_buffer, policy_module):
        """Update curriculum and track the update frame."""

        stage = {}

        if self.instance_list:
            stage["instance"] = _select_random_instance(self.instance_list)

        if self.number_agvs_type == "random":
            _min, _max = _min_max_agvs(stage["instance"])
            stage["number_agvs"] = _get_random_number_agvs(_min, _max)
        self.cfg.env = OmegaConf.merge(self.cfg.env, OmegaConf.create(stage))

        training_stage = {}
        new_frames_per_batch = self._get_frames_per_batch(stage["instance"])
        training_stage["frames_per_batch"] = new_frames_per_batch
        if self.sub_batch_size_1:
            training_stage["sub_batch_size"] = new_frames_per_batch
        self.cfg.training = OmegaConf.merge(
            self.cfg.training, OmegaConf.create(training_stage)
        )

        if env is not None:
            env.close()
            del env
        if collector is not None:
            collector.shutdown()
            del collector
        if replay_buffer is not None:
            del replay_buffer
        if self.dispatcher is None:
            new_env, new_collector, new_replay_buffer = self._update_jssp_env(
                policy_module, frames
            )
        elif self.dispatcher is not None:
            self.dispatcher.cfg = self.cfg
            new_env, new_collector, new_replay_buffer = self._update_agv_env(
                policy_module, frames
            )
        else:
            raise ValueError("Dispatcher must be provided for AGV curriculum updates.")

        self.last_update_frame = frames

        self._set_update_interval(new_frames_per_batch)

        return new_env, new_collector, new_replay_buffer

    def _set_update_interval(self, new_frames_per_batch):
        """Set the update interval based on environment instance size."""
        self.update_interval = new_frames_per_batch * self.nr_repeat_rollout


class DynamicInterval_CurriculumManager(Interval_CurriculumManager):
    def __init__(self, cfg, dispatcher: DispatcherBase = None):
        super().__init__(cfg, dispatcher)
        self.update_instance_number = self.cfg.curriculum.interval.get(
            "update_instance_number", None
        )

    def _set_update_interval(self, env):
        """Set the update interval based on environment instance size."""
        number_operations = env.num_operations
        self.update_interval = int(number_operations * self.update_instance_number)
        print(
            f"[Curriculum] Updated interval to {self.update_interval} frames; number_operations {number_operations} X update_instance_number {self.update_instance_number}."
        )


def _instance_num_operations(instance):
    storage = {
        "jssp_instances/transport/ft06": 6 * 6 + 6,
        "jssp_instances/transport/ft10": 10 * 10 + 10,
        "jssp_instances/transport/ft20": 20 * 5 + 20,
        "jssp_instances/transport/la24": 15 * 10 + 15,
        "jssp_instances/transport/la32": 30 * 10 + 30,
    }

    return storage[instance]


def _min_max_agvs(instance):
    storage = {
        "jssp_instances/transport/ft06": {"min": 3, "max": 6},
        "jssp_instances/transport/ft10": {"min": 3, "max": 10},
        "jssp_instances/transport/ft20": {"min": 3, "max": 20},
        "jssp_instances/transport/la24": {"min": 3, "max": 15},
        "jssp_instances/transport/la32": {"min": 3, "max": 30},
    }

    return storage[instance]["min"], storage[instance]["max"]


def _get_random_number_agvs(min_agvs=3, max_agvs=15):
    return random.randint(min_agvs, max_agvs)


def _select_random_instance(instance_list):
    if len(instance_list) == 0:
        return None
    return random.choice(instance_list)


def get_agv_curriculum_manager(cfg, dispatcher: DispatcherBase = None):
    """
    Get the appropriate curriculum manager based on config.

    Returns None_CurriculumManager if no curriculum config exists,
    Interval_CurriculumManager if interval config is present,
    otherwise Stages_CurriculumManager.
    """

    if hasattr(cfg.curriculum, "interval") and cfg.curriculum.interval is not None:
        interval_frames = cfg.curriculum.interval.get("update_interval_frames", None)
        interval_type = cfg.curriculum.interval.get("type", None)
        if interval_frames is not None and interval_type is not None:
            if interval_type == "static":
                return Interval_CurriculumManager(cfg, dispatcher)
            elif (
                interval_type == "dynamic"
                and cfg.curriculum.interval.get("update_instance_number", None)
                is not None
            ):
                return DynamicInterval_CurriculumManager(cfg, dispatcher)

        elif interval_frames is None and interval_type == "number":
            return NrInstances_CurriculumManager(cfg, dispatcher)

    else:
        return get_curriculum_manager(cfg, dispatcher)
