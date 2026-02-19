import random

import numpy as np

from jssp_core.domain.domains import ItemDataType
from jssp_core.instances import JSSPInstance
from jssp_core.schedule import Schedule
from jssp_core.solver.base import Heuristic


heuristic_rng = random.Random(42)

# SPT MWKR FDD/WKR MOPNR -> Transport Instances


# --- generic driver for job-only JSSP schedules ---
def _solve_job_only_with_step(instance: JSSPInstance, step_fn) -> Schedule:
    """
    Drives a Schedule using a job-selection step function.
    Assumes schedule.schedule_job(job_id) advances the schedule by one operation
    for that job and Schedule knows the next operation internally.
    """
    schedule = Schedule(instance)
    while not schedule.is_complete():
        job_id = step_fn(schedule)
        schedule.schedule_job(job_id)
    return schedule


class ScheduleProxy:
    """
    A temporary view of the schedule that 'lies' about which operations are eligible.
    It passes all other attribute access requests directly to the real schedule.
    """

    def __init__(self, real_schedule, filtered_ops):
        self._real_schedule = real_schedule
        self._filtered_ops = filtered_ops

    def get_eligible_operations(self):
        """Override to return only our non-delay filtered operations"""
        return self._filtered_ops

    def __getattr__(self, name):
        """Pass any other method calls (get_earliest_start_time, instance, etc.) to the real schedule"""
        return getattr(self._real_schedule, name)


class RandomStepHeuristic(Heuristic):
    """
    A wrapper that picks a random action every nth step.
    """

    def __init__(self, base_heuristic: Heuristic, n: int):
        self.base_heuristic = base_heuristic
        self.n = n
        self.step_count = 0
        self.item_type = base_heuristic.item_type

    def step(self, current_schedule) -> int:
        self.step_count += 1
        if self.step_count % self.n == 0:
            # Random action
            eligible_ops = current_schedule.get_eligible_operations()
            if not eligible_ops:
                raise ValueError("No eligible operations")

            # eligible_ops keys are (job_id, op_id)
            random_key = random.choice(list(eligible_ops.keys()))
            return random_key[0]  # Return job_id

        return self.base_heuristic.step(current_schedule)

    def solve(self, instance: JSSPInstance) -> Schedule:
        self.step_count = 0
        return super().solve(instance)

    def reset_count(self):
        self.step_count = 0


# ===========================
# Heuristic Implementations for Jobs
# ===========================


class SPT(Heuristic):
    """
    Shortest Processing Time (SPT) heuristic
    Always schedules the operation with the shortest duration among available operations
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        process_durations = {}
        for (job_id, op_id), _ in eligible_ops.items():
            machine, duration = current_schedule.instance[job_id][op_id]
            process_durations[(job_id, op_id)] = duration
            if not process_durations:
                raise ValueError("No eligible operations")

        min_key = min(process_durations, key=process_durations.get)
        # print("Min Key:", min_key)
        # Find operation with shortest processing time

        job_key = min_key[0]
        op_key = min_key[1]
        return job_key  # , op_key  # Return job_id


class SMPT(Heuristic):
    """
    Shortest Machine Processing Time (SMPT) heuristic
    Always schedules the operation with the shortest duration among available operations
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        least_duration_jobs = list()
        # Look through all machines
        for machine_id in range(current_schedule.num_machines):
            process_durations = {}
            # Check which jobs can be scheduled in the machine
            for (job_id, op_id), _ in eligible_ops.items():
                if current_schedule.is_job_schedulable_on_machine(job_id, machine_id):
                    _, duration = current_schedule.instance[job_id][op_id]
                    process_durations[(job_id, op_id)] = duration
            if not process_durations:
                continue
            min_key = min(process_durations, key=process_durations.get)
            least_duration_jobs.append(min_key)

        # Now select the job with the earliest starting time
        earliest_starting_time = float("inf")
        earliest_job_key = -1
        for min_key in least_duration_jobs:
            start_time = current_schedule.get_earliest_start_time(
                min_key[0], min_key[1]
            )
            if start_time < earliest_starting_time:
                earliest_job_key = min_key[0]
                earliest_starting_time = start_time

        return earliest_job_key


class LPT(Heuristic):
    """
    Longest Processing Time (LPT) heuristic
    Always schedules the operation with the longest duration among available operations
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        process_durations = {}
        for (job_id, op_id), _ in eligible_ops.items():
            machine, duration = current_schedule.instance[job_id][op_id]
            process_durations[(job_id, op_id)] = duration
            if not process_durations:
                raise ValueError("No eligible operations")

        min_key = max(process_durations, key=process_durations.get)
        # print("Max Key:", min_key)
        # Find operation with shortest processing time

        job_key = min_key[0]
        op_key = min_key[1]
        return job_key  # , op_key  # Return job_id


class MWR(Heuristic):
    """
    Most Work Remaining (MWR) heuristic
    Prioritizes jobs with the most total processing time remaining
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        remaining_work_dict = {}
        for (job_id, op_id), _ in eligible_ops.items():
            # Calculate remaining work for this job
            remaining_work = sum(
                d for _, d in current_schedule.instance[job_id][op_id:]
            )
            remaining_work_dict[job_id] = remaining_work

        job_key = max(remaining_work_dict, key=remaining_work_dict.get)
        return job_key  # , op_key  # Return job_id


class LWR(Heuristic):
    """
    Least Work Remaining (LWR) heuristic
    Prioritizes jobs with the least total processing time remaining
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        remaining_work_dict = {}
        for (job_id, op_id), _ in eligible_ops.items():
            # Calculate remaining work for this job
            remaining_work = sum(
                d for _, d in current_schedule.instance[job_id][op_id:]
            )
            remaining_work_dict[job_id] = remaining_work
        # print(remaining_work_dict)
        job_key = min(remaining_work_dict, key=remaining_work_dict.get)
        return job_key  # , op_key  # Return job_id


class FDDMWR(Heuristic):
    """
    Flow Due Date/ Most Work Remaining (FDD/MWR) Heuristic
    Based on MWR, also takes into account the "urgency" of the job
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()

        fdd_mwr_ratios = {}
        for (job_id, op_id), _ in eligible_ops.items():
            earliest_start_time = current_schedule.get_earliest_start_time(
                job_id, op_id
            )
            # Calculate remaining work for this job
            remaining_work = sum(
                d for _, d in current_schedule.instance[job_id][op_id:]
            )

            fdd = earliest_start_time + remaining_work

            ratio = fdd / (remaining_work + 1e-9)

            fdd_mwr_ratios[job_id] = ratio

        job_key = min(fdd_mwr_ratios, key=fdd_mwr_ratios.get)
        return job_key


class MOR(Heuristic):
    """
    Most Operations Remaining (MOR) heuristic
    Prioritizes jobs with the most operations remaining
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        operations_remaining_dict = {}
        for (job_id, op_id), _ in eligible_ops.items():
            # Calculate operations remaining for this job
            operations_remaining = len(current_schedule.instance[job_id]) - op_id
            operations_remaining_dict[job_id] = operations_remaining

        job_key = max(operations_remaining_dict, key=operations_remaining_dict.get)
        return job_key  # Return job_id


class LOR(Heuristic):
    """
    Least Operations Remaining (LOR) heuristic
    Prioritizes jobs with the least operations remaining
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        operations_remaining_dict = {}
        for (job_id, op_id), _ in eligible_ops.items():
            # Calculate operations remaining for this job
            operations_remaining = len(current_schedule.instance[job_id]) - op_id
            operations_remaining_dict[job_id] = operations_remaining

        job_key = min(operations_remaining_dict, key=operations_remaining_dict.get)
        return job_key  # Return job_id


class RandomJob(Heuristic):
    """
    Random Choice heuristic
    Randomly selects an eligible job to schedule
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        job_ids = list({job_id for (job_id, op_id) in eligible_ops.keys()})
        job_key = heuristic_rng.choice(job_ids)
        return job_key  # Return job_id


class FCFS(Heuristic):
    """
    First Come First Serve (FCFS) heuristic
    Processes jobs in order 0, 1, 2, ...
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        if not eligible_ops:
            raise ValueError("No eligible operations")
        first_pair = next(iter(eligible_ops))
        return first_pair[0]


# ===========================
# Heuristic Implementations for AGVs
# ===========================


# SCTA SCPT
class RandomAGV(Heuristic):
    """
    Random Choice heuristic
    Randomly selects an eligible job to schedule
    """

    item_type = ItemDataType.AGV

    def solve(self, instance: JSSPInstance) -> Schedule:
        raise NotImplementedError(
            "AGV heuristics must be used with a job heuristic via TransportSchedule"
        )

    def step(self, current_schedule, job) -> int:
        agv_id = heuristic_rng.choice(list(current_schedule.agvs.keys()))
        return agv_id  # Return job_id


class SPUT(Heuristic):
    """
    Smallest Pick Up Time heuristic.
    Pick up time= delta of agvs_ next location to pick up op location -> Travel time

    """

    item_type = ItemDataType.AGV

    def solve(self, instance: JSSPInstance) -> Schedule:
        raise NotImplementedError(
            "AGV heuristics must be used with a job heuristic via TransportSchedule"
        )

    def step(self, schedule, job) -> int:
        """Shortest Pick Up Time of all eligible AGVs -> Select AGV ID"""
        machine_to, duration = schedule.instance[job][schedule.job_next_op[job]]

        possible_arrival_time_for_agvs_2_all_locations = (
            schedule.feature_possible_arrival_time_for_agvs_2_all_locations()
        )
        agvs_to_machine_to = possible_arrival_time_for_agvs_2_all_locations[
            :, machine_to
        ]
        agv_id = int(np.argmin(agvs_to_machine_to))
        return agv_id


class SCTA(Heuristic):
    """
    Shortest Completion Time of intransport tasks of AGV
    SCTA = Completion Time of the intransport tasks of AGV
    """

    item_type = ItemDataType.AGV

    def solve(self, instance: JSSPInstance) -> Schedule:
        raise NotImplementedError(
            "AGV heuristics must be used with a job heuristic via TransportSchedule"
        )

    def step(self, schedule, job) -> int:
        """Shortest Completion Time of all eligible AGVs -> Select AGV ID"""
        agv_completion_times = []
        for agv_id, agv in schedule.agvs.items():
            completion_time = agv.available_time
            agv_completion_times.append(completion_time)
        agv_id = int(np.argmin(agv_completion_times))
        return agv_id


class SCPT(Heuristic):
    """
    Shortest Completion Time of intransport tasks + Pick Up Time of AGV
    """

    item_type = ItemDataType.AGV

    def solve(self, instance: JSSPInstance) -> Schedule:
        raise NotImplementedError(
            "AGV heuristics must be used with a job heuristic via TransportSchedule"
        )

    def step(self, schedule, job) -> int:
        # Pick Up Times for each AGV based on the job
        machine_to, duration = schedule.instance[job][schedule.job_next_op[job]]

        possible_arrival_time_for_agvs_2_all_locations = (
            schedule.feature_possible_arrival_time_for_agvs_2_all_locations()
        )
        agvs_to_machine_to = possible_arrival_time_for_agvs_2_all_locations[
            :, machine_to
        ]
        # Calculate Completion Times for each AGV
        agv_completion_times = []
        for agv_id, agv in schedule.agvs.items():
            completion_time = agv.available_time
            agv_completion_times.append(completion_time)

        combined = np.array(agvs_to_machine_to) + np.array(agv_completion_times)
        agv_id = int(np.argmin(combined))
        return agv_id


JOB_HEURISTICS_REGISTRY = {
    "spt": SPT,
    "smpt": SMPT,
    "lpt": LPT,
    "mwr": MWR,
    "lwr": LWR,
    "fddmwr": FDDMWR,
    "mor": MOR,
    "lor": LOR,
    "random": RandomJob,
    "fcfs": FCFS,
}


def job_heuristic_factory(name: str, random_nth_step: int | None = None) -> Heuristic:
    """
    Factory method to create job heuristic instances.
    """
    name = name.lower().strip()

    base_name = name

    if base_name not in JOB_HEURISTICS_REGISTRY:
        raise ValueError(f"Unknown job heuristic: {base_name} (derived from '{name}')")

    # 1. Instantiate the base heuristic (e.g., SPT())
    heuristic_instance = JOB_HEURISTICS_REGISTRY[base_name]()

    if random_nth_step is not None and random_nth_step > 0:
        heuristic_instance = RandomStepHeuristic(heuristic_instance, random_nth_step)

    return heuristic_instance


AGV_HEURISITCS = {
    "sput": SPUT,
    "random": RandomAGV,
    "scta": SCTA,
    "scpt": SCPT,
}


def agv_heuristic_factory(name: str) -> Heuristic:
    """Factory method to create agv heuristic instances by name."""

    if name not in AGV_HEURISITCS:
        raise ValueError(f"Unknown agv heuristic: {name}")
    return AGV_HEURISITCS[name]()


# ===========================
# Test all combinations of heuristics
# ===========================

# Use late imports or functions to avoid circularity if needed,
# though these are primarily for testing/benchmarking scripts.


def solve_instance_with_transport_heuristics(
    job_heuristic: Heuristic,
    agv_heuristic: Heuristic,
    transport_schedule: "TransportSchedule",
) -> tuple[int, Heuristic, Heuristic]:
    done = transport_schedule.is_complete()
    while not done:
        job_id = job_heuristic.step(transport_schedule)
        agv_id = agv_heuristic.step(transport_schedule, job_id)
        transport_schedule.schedule_job(job_id, **{"agv": agv_id})
        done = transport_schedule.is_complete()

    return (transport_schedule.get_makespan(), job_heuristic, agv_heuristic)


def solve_jobinstance_with_heuristics(
    job_heuristic: Heuristic,
    schedule: Schedule,
) -> tuple[int, Heuristic]:
    done = schedule.is_complete()
    while not done:
        job_id = job_heuristic.step(schedule)
        schedule.schedule_job(job_id)
        done = schedule.is_complete()

    return (schedule.get_makespan(), job_heuristic)


def run_all_transport_instance(file_path: str = "jssp_instances/transport/ft06.yaml"):
    from jssp_core.instances import _load_transport_instance
    from jssp_core.schedule import TransportSchedule

    storage = {}
    for job_name in JOB_HEURISTICS_REGISTRY.keys():
        for agv_name in AGV_HEURISITCS.keys():
            job_heuristic = job_heuristic_factory(job_name)
            agv_heuristic = agv_heuristic_factory(agv_name)

            instance = _load_transport_instance(file_path)
            transport_schedule = TransportSchedule(instance, num_agvs=3)

            done = transport_schedule.is_complete()
            while not done:
                job_id = job_heuristic.step(transport_schedule)
                agv_id = agv_heuristic.step(transport_schedule, job_id)
                transport_schedule.schedule_job(job_id, **{"agv": agv_id})
                done = transport_schedule.is_complete()

            storage[(job_name, agv_name)] = transport_schedule.get_makespan()

    for k, i in storage.items():
        print(f"Job Heuristic: {k[0]}, AGV Heuristic: {k[1]} => Makespan: {i}")

    min_pair = min(storage.items(), key=lambda kv: kv[1])
    print("==========================")
    print(f"Best heuristic pair: {min_pair[0]} -> Makespan: {min_pair[1]}")
    return min_pair, storage


def run_all_schedule_instance():
    from jssp_core.instances import get_ft06_instance
    from jssp_core.schedule import Schedule

    storage = {}
    for job_name in JOB_HEURISTICS_REGISTRY.keys():
        job_heuristic = job_heuristic_factory(job_name)

        instance = get_ft06_instance()
        schedule = Schedule(instance)

        done = schedule.is_complete()
        while not done:
            job_id = job_heuristic.step(schedule)
            schedule.schedule_job(int(job_id))
            done = schedule.is_complete()

        storage[(job_name)] = schedule.get_makespan()

    for k, i in storage.items():
        print(f"Job Heuristic: {k} => Makespan: {i}")

    min_pair = min(storage.items(), key=lambda kv: kv[1])
    print("==========================")
    print(f"Best heuristic: {min_pair[0]} -> Makespan: {min_pair[1]}")
    return min_pair


def get_all_job_agv_heuristic_combinations():
    combinations = []
    for job_name in JOB_HEURISTICS_REGISTRY.keys():
        for agv_name in AGV_HEURISITCS.keys():
            job_heuristic = job_heuristic_factory(job_name)
            agv_heuristic = agv_heuristic_factory(agv_name)
            combinations.append((job_heuristic, agv_heuristic))
    return combinations


if __name__ == "__main__":
    h = job_heuristic_factory("spt")
    print(h)
