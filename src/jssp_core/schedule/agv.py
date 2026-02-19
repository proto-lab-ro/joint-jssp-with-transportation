import random
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from jssp_core.schedule.jssp import Schedule


@dataclass
class AGV:
    """Simple AGV representation"""

    agv_id: int
    available_time: float = 0.0
    next_available_location: int | None = None  # machine id

    def __repr__(self):
        return f"AGV{self.agv_id}(available_time={self.available_time},next_available_location={self.next_available_location})"

    def set_scheudled_operation(self, avail_time: float, next_location: int):
        self.set_available_time(avail_time)
        self.set_next_available_location(next_location)

    def set_available_time(self, time: float):
        self.available_time = time

    def set_next_available_location(self, location: int):
        self.next_available_location = location


class TransportSchedule(Schedule):
    """
    Transport-aware scheduler with AGVs and machine input/output buffers.

    - AGVs move jobs between machines according to a provided travel_time matrix
      or a default distance-based metric.
    - Jobs' first operations are enqueued into their machines' input buffers on reset
      (arrival_time=0) to enforce FCFS start ordering.
    - When an operation finishes, a transport request is immediately scheduled
      for the next operation (if it uses a different machine). The selected AGV's
      availability is updated accordingly and the job is enqueued into the
      destination machine input buffer with the expected arrival time.
    """

    def __init__(
        self,
        instance: list[list[list[tuple[int, int]]]],  # List
        num_agvs: int = 0,
    ):
        # travel_time_matrix: shape [num_machines, num_machines]
        self.travel_time = instance[1]  # 8 x 8
        self.num_agvs = num_agvs
        self.agvs: dict[int, AGV] = {}
        self.scheduled_agvs = []  # (agv_id, start_time, duration,(job_id, op_id)) -> for Gantt chart

        if num_agvs == 0:
            self.num_agvs = sum(len(job) for job in instance[0])  # Num_operations

        super().__init__(instance[0])

    def _init_agvs(self):
        self.agvs = {
            i: AGV(
                i,
                available_time=0.0,
                next_available_location=self.id_agv_start_location,
            )
            for i in range(self.num_agvs)
        }

    def reset(self):
        """Reset schedule and also clear buffers and AGV state."""
        super().reset()
        self.num_machines = self.num_machines
        self.id_agv_start_location = self.num_machines - 2
        self.id_agv_end_location = self.id_agv_start_location + 1
        self.scheduled_agvs = []
        self.selected_job_aec = None

        self._init_agvs()

    def get_last_scheduled_machine(self) -> tuple[bool, int]:
        """Get the machine id of the last scheduled operation's machine."""
        status, aec_selected_job = self.get_selected_job_aec()
        if not status:
            return status, aec_selected_job

        job_next_info = self.get_next_operation_info(aec_selected_job)
        return True, job_next_info.machine

    def get_current_processing_machine(self):
        """Get the machine id of the last scheduled operation's machine."""
        status, aec_selected_job = self.get_selected_job_aec()
        if not status:
            return status, aec_selected_job

        job_current_info = self.get_current_operation_info(aec_selected_job)

        if job_current_info is None:
            job_current_machine = self.id_agv_start_location
        else:
            job_current_machine = job_current_info.machine
        return True, job_current_machine

    def set_travel_time_matrix(self, matrix: np.ndarray):
        assert matrix.shape == (self.num_machines, self.num_machines)
        self.travel_time_matrix = matrix

    def set_selected_job_aec(self, job_id: int):
        self.selected_job_aec = job_id

    def _get_selected_job_aec(self) -> int | None:
        return self.selected_job_aec

    def get_selected_job_aec(self) -> int | None:
        job_id = self._get_selected_job_aec()

        if job_id is None:
            return False, -1

        if self.is_complete():
            return False, -1

        if not self.can_schedule_job(job_id):
            return False, -1

        return True, job_id

    def add_agv(
        self, agv_id: int, available_time: float = 0.0, location: int | None = None
    ):
        self.agvs[agv_id] = AGV(
            agv_id, available_time=available_time, next_available_location=location
        )

    def _get_random_agv(self):
        """Return a random AGV object from self.agvs.

        Raises:
            RuntimeError: if no AGVs are defined.

        Note: returns the AGV instance (not its id).
        """
        if not self.agvs:
            raise RuntimeError("No AGVs available to select")
        return random.choice(list(self.agvs.values()))

    def _get_travel_time(self, from_location: int, to_location: int) -> float:
        if from_location == to_location:
            return 0.0
        if self.travel_time is not None:
            return float(self.travel_time[from_location][to_location])

        else:
            raise NotImplementedError("Travel transport information are missing.")

    def feature_possible_arrival_time_for_agvs_2_all_locations(self) -> np.ndarray:
        """
        Create feature matrix of shape [num_agvs, num_machines+2] with each AGV's
        available time to reach each machine location.
        Feature: available_time + travel_time to location
        """

        feature = np.zeros((self.num_agvs, self.num_machines), dtype=np.float32)
        for agv_id, agv in self.agvs.items():
            for loc in range(self.num_machines):
                travel_time = self._get_travel_time(agv.next_available_location, loc)
                feature[agv_id][loc] = agv.available_time + travel_time
        return feature

    def get_travel_time_matrix(self) -> np.ndarray:
        """
        Create feature matrix of shape [num_agvs, num_machines+2] with each AGV's
        travel time to reach each machine location.
        feature: travel_time to location
        """
        feature = np.zeros((self.num_agvs, self.num_machines), dtype=np.float32)
        for agv_id, agv in self.agvs.items():
            for loc in range(self.num_machines):
                travel_time = self._get_travel_time(agv.next_available_location, loc)
                feature[agv_id][loc] = travel_time
        return feature

    def norm_feature_possible_arrival_time_for_agvs_2_all_locations(self):
        """
        Create normalized feature matrix of shape [num_agvs, num_machines+2] with each AGV's
        available time to reach each machine location.
        Feature: available_time + travel_time to location
        """
        features = self.feature_possible_arrival_time_for_agvs_2_all_locations()
        max_col = np.max(features, axis=0)
        min_col = np.min(features, axis=0)
        normalized_features = (features - min_col) / (max_col - min_col + 1e-8)

        return normalized_features

    def feature_available_time_for_agvs(self) -> np.ndarray:
        """
        Create feature matrix of shape [num_agvs, num_machines] with each AGV's
        available time to reach each machine location.
        """
        feature = np.array(
            [agv.available_time for agv in self.agvs.values()], dtype=np.float32
        )
        return feature

    def norm_feature_available_time_for_agvs(self) -> np.ndarray:
        """
        Normalizes the feature matrix with (num_agvs ) shape by
        their min and max.
        Returns a normalized feature matrix of the same shape.
        """
        features = self.feature_available_time_for_agvs()
        max_val = np.max(features)
        min_val = np.min(features)
        normalized_features = (features - min_val) / (max_val - min_val + 1e-8)
        return normalized_features

    def _select_agv_for_transport(self, agv: int = None) -> tuple[AGV, float, float]:
        """
        Choose the AGV that achieves the earliest arrival time for a transport
        request starting no earlier than earliest_start. Returns (agv, arrival_time, travel_time).

        Args:
            agv: Optional AGV instance to use. If None, selects a random AGV.

        Returns:
            Tuple of (selected_agv, arrival_time_at_from_location, travel_time_to_from_location)
        """
        # Use provided AGV or select a random one
        selected_agv = self.agvs[agv] if agv is not None else self._get_random_agv()

        return selected_agv

    def _get_start_of_agv_from_from_location(
        self, job_id: int, arrival_time_at_from_location: float
    ) -> float:
        previous_operation_end_time = self.job_ready_time[job_id]
        waiting_time = 0.0
        if previous_operation_end_time > arrival_time_at_from_location:
            earliest_start_from_from_2_to = previous_operation_end_time
            waiting_time = previous_operation_end_time - arrival_time_at_from_location
        else:
            earliest_start_from_from_2_to = arrival_time_at_from_location

        return earliest_start_from_from_2_to, waiting_time

    def _get_machine_of_previous_operation(self, job_id: int, op_id: int) -> int | None:
        if op_id > 0:
            from_machine, _ = self.instance[job_id][op_id - 1]
            return from_machine
        else:
            return self.id_agv_start_location

    def schedule_job(self, job_id: int, **kwargs: dict) -> bool:
        """
        Schedule next operation for a job, including transport handling.
        This overrides parent to enforce buffer/AGV constraints.
        """
        try:
            job_id = int(job_id.item())
        except Exception:
            job_id = int(job_id)

        if not self.can_schedule_job(job_id):
            return False

        op_id = self.job_next_op[job_id]

        if op_id >= len(self.instance[job_id]):
            return False

        machine, duration = self.instance[job_id][op_id]
        from_machine = self._get_machine_of_previous_operation(job_id, op_id)  # 5,3

        agv = self._select_agv_for_transport(agv=kwargs.get("agv", None))
        travel_time_2_from_location = self._get_travel_time(
            agv.next_available_location, from_machine
        )
        arrival_time_at_from_location = agv.available_time + travel_time_2_from_location

        (
            earliest_start_from_from_2_to,
            waiting_time,
        ) = self._get_start_of_agv_from_from_location(
            job_id, arrival_time_at_from_location
        )
        agv_travel_time_from_2_to_location = self._get_travel_time(
            from_machine, machine
        )
        arrival_time = (
            earliest_start_from_from_2_to + agv_travel_time_from_2_to_location
        )
        complete_agv_travel_time = arrival_time - agv.available_time

        if machine == self.id_agv_end_location:
            start_time = arrival_time

        else:
            start_time = max(self.machine_ready_time.get(machine, 0.0), arrival_time)
        end_time = start_time + duration

        self.scheduled[(job_id, op_id)] = start_time
        self.job_next_op[job_id] += 1
        self.job_ready_time[job_id] = end_time

        mk = self.get_makespan()
        d = self.is_complete()

        if machine == self.id_agv_end_location:
            pass
        else:
            self.machine_ready_time[machine] = end_time

        agv_time_dict = {
            "available_time": agv.available_time,
            "travel_current_2_from_location_duration": travel_time_2_from_location,
            "arrival_time_at_from_location": arrival_time_at_from_location,
            "waiting_for_from_location_finish_duration": waiting_time,
            "earliest_start_from_from_2_to": earliest_start_from_from_2_to,
            "travel_from_2_to_location_duration": agv_travel_time_from_2_to_location,
            "complete_agv_travel_time": complete_agv_travel_time,
            "arrival_time_at_to_location": arrival_time,
        }
        self.scheduled_agvs.append(
            (
                agv.agv_id,
                agv.available_time,
                complete_agv_travel_time,
                agv_time_dict,
                (job_id, op_id),
            )
        )
        agv.set_scheudled_operation(avail_time=arrival_time, next_location=machine)

        self.eligible_operations[(job_id, op_id)] = 0
        if op_id + 1 < len(self.instance[job_id]):
            self.eligible_operations[(job_id, op_id + 1)] = 1

        return True

    def get_agv_states(self) -> dict[int, dict]:
        return {
            aid: {
                "available_time": agv.available_time,
                "next_available_location": agv.next_available_location,
            }
            for aid, agv in self.agvs.items()
        }

    def get_lower_bound_makespan(self) -> float:
        """
        Calculate the lower bound makespan for the current schedule state.

        The lower bound is calculated as the maximum of:
        1. Job-based lower bound: For each job, current completion time + remaining processing time
        2. Machine-based lower bound: For each machine, current completion time + remaining processing time
        3. Critical path lower bound: For each unscheduled operation, the critical longest path

        Returns:
            float: Lower bound estimate for makespan
        """
        if self.is_complete():
            return self.get_makespan()

        lower_bound = 0.0

        # 1. Job-based lower bound
        for job_id in range(self.num_jobs):
            job_lower_bound = self.job_ready_time[job_id]

            # Add remaining processing time for this job
            for op_id in range(self.job_next_op[job_id], len(self.instance[job_id])):
                to_m, duration = self.instance[job_id][op_id]
                from_m = self._get_machine_of_previous_operation(job_id, op_id)
                travel_time = self._get_travel_time(from_m, to_m)

                job_lower_bound += duration + travel_time

            lower_bound = max(lower_bound, job_lower_bound)

        # 2. Machine-based lower bound
        machine_remaining_work = [0.0] * self.num_machines

        # Calculate remaining work for each machine
        for job_id in range(self.num_jobs):
            for op_id in range(self.job_next_op[job_id], len(self.instance[job_id])):
                machine, duration = self.instance[job_id][op_id]
                machine_remaining_work[machine] += duration

        for machine_id in range(self.num_machines):
            machine_ready_time = self.machine_ready_time.get(machine_id, 0.0)
            machine_lower_bound = (
                machine_ready_time + machine_remaining_work[machine_id]
            )
            lower_bound = max(lower_bound, machine_lower_bound)

        # 3. Critical path lower bound for each unscheduled operation
        for job_id in range(self.num_jobs):
            for op_id in range(self.job_next_op[job_id], len(self.instance[job_id])):
                op_lower_bound = self._calculate_operation_critical_path_lower_bound(
                    job_id, op_id
                )
                lower_bound = max(lower_bound, op_lower_bound)

        return lower_bound

    def get_maximal_bound_makespan(self) -> float:
        """
        Calculate the maximal bound makespan for the current schedule state.

        The maximal bound is calculated as the sum of:
        - Current makespan
        - Total remaining processing time for all unscheduled operations

        Returns:
            float: Maximal bound estimate for makespan
        """
        if self.is_complete():
            return self.get_makespan()

        current_makespan = self.get_makespan()
        total_remaining_time = 0.0

        # Sum remaining processing time for all unscheduled operations
        for job_id in range(self.num_jobs):
            for op_id in range(self.job_next_op[job_id], len(self.instance[job_id])):
                to_m, duration = self.instance[job_id][op_id]
                from_m = self._get_machine_of_previous_operation(job_id, op_id)
                travel_time = self._get_travel_time(from_m, to_m)
                total_remaining_time += duration + travel_time

        maximal_bound = current_makespan + total_remaining_time
        return maximal_bound

    def _validate_agv_allocations(self) -> tuple[bool, list[str]]:
        """Validate AGV allocations recorded in self.scheduled_agvs.

        Checks performed:
        - Structural validation for each scheduled_agvs entry supporting both legacy
          and new formats. For the new format entries include an agv_time_dict with
          timing info and a final (job_id, op_id) tuple.

        Returns:
            (is_valid: bool, violations: List[str])
        """
        violations: list[str] = []
        agv_entries: dict[int, list[dict]] = defaultdict(list)

        for idx, entry in enumerate(self.scheduled_agvs):
            # Support multiple legacy/new shapes. Goal: extract agv_id, optional timing dict, and job tuple.
            agv_id = None
            jobop = None
            agv_time_dict = None

            # entry might be a tuple/list
            if not isinstance(entry, (list, tuple)):
                violations.append(
                    f"scheduled_agvs[{idx}] has invalid type: {type(entry)}"
                )
                continue

            if len(entry) >= 5 and isinstance(entry[3], dict):
                try:
                    agv_id = int(entry[0])
                    agv_time_dict = dict(entry[3])
                    jobop = entry[4]
                except Exception:
                    violations.append(
                        f"scheduled_agvs[{idx}] has malformed new-format entry: {entry}"
                    )
                    continue
            else:
                try:
                    agv_id = int(entry[0])
                except Exception:
                    violations.append(
                        f"scheduled_agvs[{idx}] missing agv id in first position: {entry}"
                    )
                    continue

                candidate = entry[-1]
                if (
                    isinstance(candidate, tuple)
                    and len(candidate) == 2
                    and all(isinstance(x, int) for x in candidate)
                ):
                    jobop = candidate

                for item in entry:
                    if isinstance(item, dict):
                        agv_time_dict = dict(item)
                        break

            if agv_id is None:
                violations.append(
                    f"scheduled_agvs[{idx}] could not determine agv_id: {entry}"
                )
                continue

            if agv_time_dict is None:
                pass

            if jobop is None:
                violations.append(
                    f"scheduled_agvs[{idx}] missing job tuple (job_id, op_id): {entry}"
                )

            agv_entries[agv_id].append(
                {"raw": entry, "jobop": jobop, "time_dict": agv_time_dict}
            )

        for aid, items in agv_entries.items():
            if aid not in self.agvs:
                violations.append(
                    f"AGV{aid} referenced in scheduled_agvs but not present in TransportSchedule.agvs"
                )
            if not items:
                violations.append(f"AGV{aid} has no scheduled entries")

        for aid, items in agv_entries.items():
            tasks_with_times = []
            for i, item in enumerate(items):
                time_dict = item.get("time_dict")
                if time_dict is None:
                    continue

                start = time_dict.get("available_time")
                end = time_dict.get("arrival_time_at_to_location")

                if start is None or end is None:
                    violations.append(
                        f"AGV{aid} task {i} missing 'available_time' or 'arrival_time_at_to_location' in time_dict"
                    )
                    continue

                try:
                    start_f = float(start)
                    end_f = float(end)
                    tasks_with_times.append((start_f, end_f, i, item.get("jobop")))
                except (ValueError, TypeError):
                    violations.append(
                        f"AGV{aid} task {i} has non-numeric timing values: start={start}, end={end}"
                    )

            tasks_with_times.sort(key=lambda x: x[0])

            for j in range(len(tasks_with_times) - 1):
                curr_start, curr_end, curr_idx, curr_job = tasks_with_times[j]
                next_start, next_end, next_idx, next_job = tasks_with_times[j + 1]

                if curr_end > next_start:
                    violations.append(
                        f"AGV{aid} overlap: task {curr_idx} (job {curr_job}) ends at {curr_end:.2f} "
                        f"but task {next_idx} (job {next_job}) starts at {next_start:.2f}"
                    )

        return (len(violations) == 0), violations

    def plot_gantt(
        self, figsize=(12, 6), show_agvs: bool = True, save_path: str | None = None
    ):
        """Plot a simple Gantt chart of scheduled operations and AGV transports.

        - Machine operations are shown on the top axes (one row per machine).
        - AGV transports (if any) are shown on a second axes (one row per AGV) when show_agvs=True.

        Args:
            figsize: Figure size tuple.
            show_agvs: Whether to draw AGV transport lanes.
            save_path: If provided, save the figure to this path.

        Returns:
            (fig, axes) tuple where axes is (ax_ops, ax_agvs) or (ax_ops, None).
        """
        if plt is None:
            raise RuntimeError(
                "matplotlib is required to plot Gantt charts. Install matplotlib and retry."
            )

        ops = []
        for (job_id, op_id), start in self.scheduled.items():
            machine, duration = self.instance[job_id][op_id]
            ops.append(
                (machine, float(start), float(duration), int(job_id), int(op_id))
            )

        agv_items = []
        for entry in self.scheduled_agvs:
            agv_id = None
            jobop = None
            time_dict = None

            if len(entry) >= 5 and isinstance(entry[3], dict):
                try:
                    agv_id = int(entry[0])
                    time_dict = entry[3]
                    jobop = entry[4]
                except Exception:
                    continue
            else:
                # Legacy format fallback
                try:
                    agv_id = int(entry[0])
                    agv_start = float(entry[1])
                    agv_dur = float(entry[2])
                    jobop = entry[3] if len(entry) > 3 else None
                    # Create minimal time_dict for legacy
                    time_dict = {
                        "available_time": agv_start,
                        "travel_current_2_from_location_duration": 0,
                        "waiting_for_from_location_finish_duration": 0,
                        "travel_from_2_to_location_duration": agv_dur,
                    }
                except Exception:
                    continue

            if agv_id is not None and time_dict is not None:
                agv_items.append((agv_id, time_dict, jobop))

        n_machines = max(1, self.num_machines)

        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
        print("...Starting Gantt plot...")
        # Set professional style
        ax.set_facecolor("#fafafa")
        ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5, color="gray")

        # Color map for jobs - using more distinct colors
        job_ids = sorted({job_id for (_, _, _, job_id, _) in ops})
        # Use a combination of tab20 and tab20b for more distinct colors
        cmap = plt.get_cmap("tab10")
        job_color = {jid: cmap(i % 10) for i, jid in enumerate(job_ids)}

        # Plot operations per machine as horizontal bars
        for machine in range(n_machines):
            # draw a subtle background band for each machine with alternating shades
            bg_color = "#ffffff" if machine % 2 == 0 else "#f5f5f5"
            ax.axhspan(
                machine - 0.5, machine + 0.5, facecolor=bg_color, alpha=0.3, zorder=0
            )
        print("...AGVs...")
        for machine, start, dur, job_id, op_id in ops:
            # Professional bars with subtle edge and shadow effect
            color = job_color.get(job_id)
            if dur == 0.0:
                print(dur)
                dur = 2  # minimal visible width for zero-duration ops
            ax.barh(
                machine,
                dur,
                left=start,
                height=0.6,
                align="center",
                color=color,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.9,
            )

            # Add text label with better contrast
            if dur > 0.5:  # Only show label if bar is wide enough
                ax.text(
                    start + dur / 2.0,
                    machine,
                    f"J{job_id}·O{op_id}",
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.3,
                    ),
                )

        # Plot AGV transports below machines if requested
        if show_agvs and agv_items:
            agv_ids = sorted({aid for (aid, _, _) in agv_items})

            for aid in agv_ids:
                # AGVs are plotted below machines with alternating background
                agv_y_position = n_machines + aid
                bg_color = "#e8f4f8" if aid % 2 == 0 else "#d4e9f0"
                ax.axhspan(
                    agv_y_position - 0.5,
                    agv_y_position + 0.5,
                    facecolor=bg_color,
                    alpha=0.4,
                    zorder=0,
                )

            # Plot color-coded segments for each AGV task
            for aid, time_dict, jobop in agv_items:
                agv_y_position = n_machines + aid

                # Extract timing segments from time_dict
                avail_time = time_dict.get("available_time", 0)
                travel_to_from_dur = time_dict.get(
                    "travel_current_2_from_location_duration", 0
                )
                waiting_dur = time_dict.get(
                    "waiting_for_from_location_finish_duration", 0
                )
                travel_to_to_dur = time_dict.get(
                    "travel_from_2_to_location_duration", 0
                )

                # Get job color if available
                job_color_val = "#808080"  # default gray
                if jobop:
                    try:
                        jid, oid = jobop
                        job_color_val = job_color.get(jid, "#808080")
                    except Exception:
                        pass

                # Segment 1: Travel to from-location (dark gray with pattern)
                if travel_to_from_dur > 0:
                    ax.barh(
                        agv_y_position,
                        travel_to_from_dur,
                        left=avail_time,
                        height=0.6,
                        align="center",
                        color="#555555",
                        edgecolor="white",
                        linewidth=1.5,
                        alpha=0.85,
                        hatch="///",
                    )

                # Segment 2: Waiting at from-location (orange/amber - more professional than red)
                waiting_start = avail_time + travel_to_from_dur
                if waiting_dur > 0:
                    ax.barh(
                        agv_y_position,
                        waiting_dur,
                        left=waiting_start,
                        height=0.6,
                        align="center",
                        color="#FF6B35",
                        edgecolor="white",
                        linewidth=1.5,
                        alpha=0.85,
                        hatch="xxx",
                    )

                # Segment 3: Travel to to-location (job color - loaded transport)
                travel_to_to_start = waiting_start + waiting_dur
                if travel_to_to_dur > 0:
                    ax.barh(
                        agv_y_position,
                        travel_to_to_dur,
                        left=travel_to_to_start,
                        height=0.6,
                        align="center",
                        color=job_color_val,
                        edgecolor="white",
                        linewidth=1.5,
                        alpha=0.9,
                    )

                # Add label in the middle of the entire task
                total_dur = travel_to_from_dur + waiting_dur + travel_to_to_dur
                mid_point = avail_time + total_dur / 2.0
                label = f"AGV{aid}"
                if jobop:
                    try:
                        jid, oid = jobop
                        label = f"J{jid}O{oid}"
                    except Exception:
                        pass

                # Only show label if there's enough space
                if total_dur > 0.8:
                    ax.text(
                        mid_point,
                        agv_y_position,
                        label,
                        va="center",
                        ha="center",
                        fontsize=8,
                        color="white",
                        weight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="black",
                            edgecolor="none",
                            alpha=0.3,
                        ),
                    )

        # Set up combined y-axis labels (machines + AGVs)
        all_yticks = list(range(n_machines))
        all_ylabels = [f"M{m}" for m in range(n_machines)]

        if show_agvs and agv_items:
            agv_ids = sorted({aid for (aid, _, _) in agv_items})
            all_yticks.extend([n_machines + aid for aid in agv_ids])
            all_ylabels.extend([f"AGV{a}" for a in agv_ids])

        ax.set_yticks(all_yticks)
        ax.set_yticklabels(all_ylabels, fontsize=10, weight="medium")
        ax.set_ylabel("Resources", fontsize=12, weight="bold", labelpad=10)
        ax.set_xlabel("Time (units)", fontsize=12, weight="bold", labelpad=10)
        ax.set_title(
            "Job Shop Schedule with Transportation", fontsize=14, weight="bold", pad=15
        )
        ax.invert_yaxis()

        # Style the axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(axis="both", which="major", labelsize=10, length=5, width=1.5)

        # Enhanced legend with AGV segment explanation
        legend_patches = []
        if job_ids:
            legend_patches.extend(
                [
                    mpatches.Patch(color=job_color[jid], label=f"Job {jid}")
                    for jid in job_ids
                ]
            )

        if show_agvs and agv_items:
            # Add separator
            if legend_patches:
                legend_patches.append(mpatches.Patch(color="none", label=""))
                legend_patches.append(
                    mpatches.Patch(color="none", label="AGV Segments:")
                )
            legend_patches.extend(
                [
                    mpatches.Patch(color="#555555", label="Empty travel", hatch="///"),
                    mpatches.Patch(color="#FF6B35", label="Waiting", hatch="xxx"),
                    mpatches.Patch(color="gray", label="Loaded travel"),
                ]
            )

        if legend_patches:
            ax.legend(
                handles=legend_patches,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=9,
                frameon=True,
                shadow=True,
                fancybox=True,
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig, ax

    def copy(self):
        """Create a deep copy of the current transport schedule"""
        new_schedule = TransportSchedule(
            [self.instance, self.travel_time], num_agvs=self.num_agvs
        )
        new_schedule.job_next_op = self.job_next_op[:]
        new_schedule.job_ready_time = self.job_ready_time[:]
        new_schedule.machine_ready_time = dict(self.machine_ready_time)
        new_schedule.scheduled = dict(self.scheduled)
        new_schedule.eligible_operations = dict(self.eligible_operations)
        new_schedule.agvs = {
            aid: AGV(agv.agv_id, agv.available_time, agv.next_available_location)
            for aid, agv in self.agvs.items()
        }
        new_schedule.scheduled_agvs = list(self.scheduled_agvs)

        return new_schedule

    def validate_schedule(self) -> tuple[bool, list[str]]:
        no_violations, violations_1 = super().validate_schedule()

        # Check AGV Violations
        no_violations_2, violations_2 = self._validate_agv_allocations()

        all_violations = violations_1 + violations_2

        return no_violations or no_violations_2, all_violations

    def _get_eligible_agvs(self):
        return list(self.agvs.keys())


def test_scheduler(schedule):
    print(f"Initial state: {schedule.get_schedule_summary()}")

    # Schedule some operations
    eligible_jobs = schedule.get_eligible_jobs()
    print(f"Initially eligible jobs: {eligible_jobs}")

    # Schedule first few operations
    for i, job_id in enumerate(eligible_jobs[:3]):
        success = schedule.schedule_operation(job_id)
        print(f"Scheduled job {job_id}: {success}")
        print(f"State after step {i + 1}: {schedule.get_schedule_summary()}", " \n")

    # Test different heuristic estimates
    print("\n=== Testing Heuristic Estimates ===")
    print(f"SPT estimate: {schedule.estimate_completion_time('SPT')}")
    print(f"LPT estimate: {schedule.estimate_completion_time('LPT')}")
    print(f"MWR estimate: {schedule.estimate_completion_time('MWR')}")
    print(f"LWR estimate: {schedule.estimate_completion_time('LWR')}")

    # Test multiple heuristics at once
    multiple_estimates = schedule.estimate_completion_time_multiple_heuristics()
    print(f"\nMultiple heuristic estimates: {multiple_estimates}")

    # Find best heuristic estimate
    best_heuristic, best_time = schedule.get_best_heuristic_estimate()
    print(f"Best heuristic: {best_heuristic} with time: {best_time}")

    # Validate schedule
    is_valid, violations = schedule.validate_schedule()
    print(f"\nSchedule is valid: {is_valid}")
    if violations:
        print(f"Violations: {violations}")

    print(f"Final makespan: {schedule.get_makespan()}")
    print(f"Gantt data entries: {len(schedule.get_gantt_data())}")

    fig, axes = schedule.plot_gantt(show_agvs=True)
    plt.show()
    return schedule


if __name__ == "__main__":
    # Test the Schedule class
    from jssp_core.instances import (
        _load_transport_instance,
    )

    print("Testing Loading Transport Instance")
    instance = _load_transport_instance()
    transport_schedule = TransportSchedule(instance)
    # tansport_schedule = test_scheduler(transport_schedule)

    print("Testing a complete schedule")
    instance = _load_transport_instance()
    transport_schedule = TransportSchedule(instance, num_agvs=3)
    scheudule_count = 1
    while not transport_schedule.is_complete():
        eligible_jobs = transport_schedule.get_eligible_jobs()
        if not eligible_jobs:
            print("No eligible jobs to schedule, stopping.")
            break
        job_id = random.choice(eligible_jobs)
        print(f"Scheduling job {job_id}")
        transport_schedule.schedule_job(job_id)
        scheudule_count += 1
    validations = transport_schedule.validate_schedule()
    print(validations)
    fig, axes = transport_schedule.plot_gantt(show_agvs=True)
    plt.show()

    print("Finished")
