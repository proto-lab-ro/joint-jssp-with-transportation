import numpy as np
import torch
from gymnasium import spaces

from jssp_core.domain import ObservationType
from jssp_core.domain.observation import ObservationData, ObservationProvider
from jssp_core.registry import OBSERVATION_REGISTRY
from jssp_core.schedule import Schedule, TransportSchedule


class BaseAgvMarlObservationProvider(ObservationProvider):
    """Base class for AGV MARL observation providers."""

    def __init__(
        self,
        schedule: TransportSchedule,
        observation_type: ObservationType = ObservationType.DICT,
    ):
        super().__init__(schedule)

        self._observation_type = observation_type
        if self._observation_type not in (ObservationType.DICT, ObservationType.FLAT):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def get_observation_space(self) -> spaces.Dict:
        obs_dict_space = spaces.Dict(
            {
                "feat": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_agvs, self.num_features),
                ),
            }
        )
        if self.observation_type == ObservationType.FLAT:
            return self.get_flattened_observation_space()

        elif self.observation_type == ObservationType.DICT:
            return obs_dict_space
        else:
            raise ValueError(
                f"Unsupported observation_type '{self.observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    def get_flattened_observation_space(self) -> spaces.Box:
        total_size = self.num_features * self.num_agvs
        return spaces.Box(low=0.0, high=1.0, shape=(total_size,), dtype=np.float32)

    def _get_next_processing_machine(self, schedule):
        """Feattuure: one hot encoding to the target machine - processing machine of next job"""
        status, next_processing_machine = schedule.get_last_scheduled_machine()
        if not status:
            feature_one_hot_encoded_selected_machine = np.zeros(
                (self.num_machines,), dtype=np.int32
            )
        else:
            feature_one_hot_encoded_selected_machine = np.zeros(
                (self.num_machines,), dtype=np.int32
            )
            feature_one_hot_encoded_selected_machine[next_processing_machine] = 1  #
        return (
            feature_one_hot_encoded_selected_machine,
            status,
            next_processing_machine,
        )

    def _get_current_processing_machine(self, schedule):
        """Feattuure: one hot encoding to the target machine - processing machine of current job"""
        status, current_processing_machine = schedule.get_current_processing_machine()
        if not status:
            feature_one_hot_encoded_selected_machine = np.zeros(
                (self.num_machines,), dtype=np.int32
            )
        else:
            feature_one_hot_encoded_selected_machine = np.zeros(
                (self.num_machines,), dtype=np.int32
            )
            feature_one_hot_encoded_selected_machine[current_processing_machine] = 1  #
        return (
            feature_one_hot_encoded_selected_machine,
            status,
            current_processing_machine,
        )

    def _select_machine_column(self, time_matrix, status, selected_loc):
        """Extract the corresponsind gcolumn for the selected machine from a matrix with
        [num.agvs, num_machines] shape
        """

        if not status:
            time_matrix_for_selected_location = np.zeros(
                (self.num_agvs, 1), dtype=np.float32
            )
        else:
            time_matrix_for_selected_location = time_matrix[:, selected_loc].reshape(
                self.num_agvs, 1
            )
        return time_matrix_for_selected_location

    def get_collection_of_feats(self):
        collection_of_feats = {
            "CTa": self.feat_CTa,
            "PUT": self.feat_PUT,
            "CTa+PUT": self.feat_CTa_PLUS_PUT,
            "TT": self.feat_TT,
            "CTm": self.feat_CTm,
            "CTj": self.feat_CTj,
        }
        return collection_of_feats

    def feat_PUT(self, schedule: TransportSchedule, selected_loc: int, status: bool):
        """Feature: Pick up Time to the selected location"""
        travel_time_matrix = schedule.get_travel_time_matrix()
        feature_put = self._select_machine_column(
            time_matrix=travel_time_matrix,
            status=status,
            selected_loc=selected_loc,
        )
        return feature_put

    def feat_CTa_PLUS_PUT(
        self, schedule: TransportSchedule, selected_loc: int, status: bool
    ):
        """Feature: Sum of Shortest Completing Time and Pick up Time to the selected location"""
        arrival_time_at_locations = (
            schedule.feature_possible_arrival_time_for_agvs_2_all_locations()
        )
        feature_CTa_plus_put = self._select_machine_column(
            time_matrix=arrival_time_at_locations,
            status=status,
            selected_loc=selected_loc,
        )
        return feature_CTa_plus_put

    def feat_CTa(self, schedule: TransportSchedule):
        """Feature: Shortest Completing Time (Available Time)"""
        CTa = schedule.norm_feature_available_time_for_agvs().reshape(
            (self.num_agvs, 1)
        )
        return CTa

    def feat_TT(
        self, schedule: TransportSchedule, from_loc: int, to_loc: int, status: bool
    ):
        """Feature: Transport time of agv to the selected machine (from_loc to to_loc) -> Same for all agvs"""
        # Alles noch abgefangen werden
        if not status:
            travel_time_between_locs = np.zeros((self.num_agvs, 1), dtype=np.float32)
            return travel_time_between_locs
        else:
            travel_time_between_locs = np.array(
                [schedule._get_travel_time(from_loc, to_loc)]
            )

            return np.repeat(travel_time_between_locs, self.num_agvs).reshape(
                self.num_agvs, 1
            )

    def feat_CTm(self, schedule: TransportSchedule, machine_id: int, status: bool):
        """Feature: Shortest completion time of the machine -> next available time of the machine"""
        if not status:
            return np.zeros((self.num_agvs, 1), dtype=np.float32)
        else:
            machine_avail_time = schedule.machine_ready_time.get(machine_id, 0.0)
            max_value = (
                max(schedule.machine_ready_time.values())
                if schedule.machine_ready_time
                else 1.0
            )
            min_value = (
                min(schedule.machine_ready_time.values())
                if schedule.machine_ready_time
                else -1.0
            )
            normalized_machine_avail_time = (
                (machine_avail_time - min_value) / (max_value - min_value)
                if max_value > min_value
                else 0.0
            )
            return np.repeat(normalized_machine_avail_time, self.num_agvs).reshape(
                self.num_agvs, 1
            )

    def feat_CTj(self, schedule: TransportSchedule):
        """Feature: Shortest completion time of the job -> next available time of the job"""
        status, job_id = schedule.get_selected_job_aec()
        if not status:
            return np.zeros((self.num_agvs, 1), dtype=np.float32)
        else:
            job_avail_time = schedule.job_ready_time[job_id]
            max_value = max(schedule.job_ready_time) if schedule.job_ready_time else 1.0
            min_value = (
                min(schedule.job_ready_time) if schedule.job_ready_time else -1.0
            )
            normalized_job_avail_time = (
                (job_avail_time - min_value) / (max_value - min_value)
                if max_value > min_value
                else 0.0
            )
            return np.repeat(normalized_job_avail_time, self.num_agvs).reshape(
                self.num_agvs, 1
            )

    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """Normalize feature to [-1, 1] range based on max value in the schedule."""

        max_value = np.max(feature) if np.max(feature) > 0 else 1.0
        min_value = np.min(feature) if np.min(feature) < 0 else -1.0
        normalized_feature = (feature - min_value) / (max_value - min_value)
        return normalized_feature


@OBSERVATION_REGISTRY.register("agv_marl_default")
class AGV_MARL(BaseAgvMarlObservationProvider):
    """Dummy provider that returns an all-ones vector as the observation."""

    def __init__(
        self,
        schedule: TransportSchedule,
        observation_type: ObservationType = ObservationType.DICT,
    ):
        super().__init__(schedule, observation_type)
        self.num_features = 6  

    @property
    def name(self) -> str:
        return "agv_marl_default"

    def get_observation(self, schedule: Schedule) -> ObservationData:
        if not isinstance(schedule, TransportSchedule):
            return {
                "feat": torch.zeros(
                    (self.num_agvs, self.num_features), dtype=torch.float32
                )
            }

        (
            feature_one_hot_encoded_selected_machine,
            status,
            next_processing_machine_id,
        ) = self._get_next_processing_machine(schedule)
        (
            _,
            status_current,
            current_processing_machine_id,
        ) = self._get_current_processing_machine(schedule)

        CTaPUT = self._normalize_feature(
            self.feat_CTa_PLUS_PUT(
                schedule,
                selected_loc=current_processing_machine_id,
                status=status_current,
            )
        )

        PUT = self._normalize_feature(
            self.feat_PUT(
                schedule,
                selected_loc=current_processing_machine_id,
                status=status_current,
            )
        )

        CTa = self.feat_CTa(schedule)
        TT = self._normalize_feature(
            self.feat_TT(
                schedule,
                from_loc=current_processing_machine_id,
                to_loc=next_processing_machine_id,
                status=status_current and status,
            )
        )

        CTaPUTandTT = self._normalize_feature(CTaPUT + TT)

        CTm = self.feat_CTm(
            schedule,
            machine_id=next_processing_machine_id,
            status=status,
        )
       
        CTj = self.feat_CTj(schedule)

        temp_obs_dict = {
            "CTa": CTa,
            "PUT": PUT,
            "CTa+PUT": CTaPUT,
            "CTaPUT+TT": CTaPUTandTT,
            "CTm": CTm,
            "CTj": CTj,
        }
        obs_dict: ObservationData = {
            "feat": torch.concatenate(
                [torch.tensor(v) for v in temp_obs_dict.values()], dim=1
            )
        }
        

        if self.observation_type == ObservationType.FLAT:
            return self.flatten_tensordict(obs_dict)

        elif self.observation_type == ObservationType.DICT:
            return obs_dict


AGV_MARL_OBSERVATION_PROVIDERS = {
    "default": OBSERVATION_REGISTRY.get("agv_marl_default"),
}


def get_agv_observation_provider(
    name: str, schedule: TransportSchedule, **kwargs
) -> ObservationProvider:
    """
    Factory function to create observation providers.

    Args:
        name: Name of the observation provider
        schedule: Schedule instance for initialization
        **kwargs: Additional arguments for specific providers

    Returns:
        ObservationProvider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    if name not in AGV_MARL_OBSERVATION_PROVIDERS:
        available = list(AGV_MARL_OBSERVATION_PROVIDERS.keys())
        raise ValueError(
            f"Unknown observation provider '{name}'. Available: {available}"
        )

    provider_class = AGV_MARL_OBSERVATION_PROVIDERS[name]
    return provider_class(schedule, **kwargs)
