import random

import yaml

from jssp_core.instances.jssp import (
    JSSPInstance,
    _get_norm_truncated_dist,
    _get_truncated_dist,
    generate_norm_truncated_normal_jssp_instance,
    generate_random_jssp_instance,
    generate_truncated_normal_jssp_instance,
    get_instance_info,
    read_yaml_specification_instance,
    read_yaml_specification_logistics,
)


class JSSPTransportInstance(list):
    """
    Represents a Job Shop Scheduling Problem instance as a list of jobs with a list of machine travel times.
    list[0] = JSSPInstance
    list[1] = travel time list

    Each job is a list of (machine, duration) tuples.
    Each machine is a list of travel times to other machines.
    In addition: start and end locations of operations are presented in the travel time,list
    """

    def __repr__(self):
        lines = []
        for job in self[0]:
            line = " ".join(f"{m} {d}" for m, d in job)
            lines.append(line)
        return (
            f"JSSPTransportInstance(num_jobs={self.num_jobs()}, num_machines={self.num_machines()}):\n"
            + "\n".join(lines)
        )

    @classmethod
    def from_file(cls, file_path: str) -> "JSSPInstance":
        """
        Create a JSSPInstance from a file.

        Args:
            file_path: Path to the instance file

        Returns:
            JSSPInstance loaded from the file
        """
        return cls(_load_transport_instance(file_path))

    def num_jobs(self) -> int:
        """Return the number of jobs in the instance."""
        return len(self[0])

    def num_machines(self) -> int:
        """Return the number of unique machines used in the instance."""
        return len(set(m for job in self[0] for m, _ in job))


def _load_transport_instance(
    file_path: str = "jssp_instances/transport/ft06.yaml",
) -> JSSPTransportInstance:
    instance = []
    with open(file_path) as f:
        instance_dict = yaml.safe_load(f)

    return _parse_transport_instance(instance_dict)


def _parse_transport_instance(instance_dict: dict) -> JSSPTransportInstance:
    """
    Parse a JSSP transport instance from a raw text string.

    Args:
        text: Raw instance text
    """
    instance_path = instance_dict["instance_config"]["instance"]
    job_instance = instance_path["specification"]
    description = instance_path["description"]
    logistics_instance = instance_dict["instance_config"]["logistics"]["specification"]

    instance_jobs = read_yaml_specification_instance(
        job_instance
    )  # [job:[(m,d), ...], ....]
    instance_travel_times = read_yaml_specification_logistics(logistics_instance)

    out_buf_idx = len(instance_travel_times) - 1
    duration = 0
    for job in instance_jobs:
        job.append((out_buf_idx, duration))
    return JSSPTransportInstance([instance_jobs, instance_travel_times])


def generate_random_transport_instance(
    num_jobs: int,
    num_machines: int,
    min_duration: int = 1,
    max_duration: int = 10,
) -> JSSPTransportInstance:
    """
    Generate a random JSSP transport instance.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        max_operation_duration: Maximum duration of an operation
        max_travel_time: Maximum travel time between machines

    Returns:
        JSSPTransportInstance
    """
    jssp_instance = generate_random_jssp_instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    travel_times = [
        [random.randint(min_duration, max_duration) for _ in range(num_machines + 2)]
        for _ in range(num_machines + 2)
    ]
    # ensure self-travel times are zero (for machines and any buffer nodes)
    for i in range(len(travel_times)):
        travel_times[i][i] = 0

    travel_times = [tuple(row) for row in travel_times]
    out_buf_idx = num_machines + 1
    duration = 0
    for job in jssp_instance:
        job.append((out_buf_idx, duration))
    return JSSPTransportInstance([jssp_instance, travel_times])


def generate_grid_random_transport_instance(
    num_jobs: int,
    num_machines: int,
    min_op_duration: int,
    max_op_duration: int,
    min_travel_duration: int,
    max_travel_duration: int,
    seed: int | None = None,
) -> JSSPTransportInstance:
    """
    Generate a random JSSP transport instance.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        min_op_duration: Minimum duration of an operation
        max_op_duration: Maximum duration of an operation
        min_travel_duration: Minimum travel duration between machines
        max_travel_duration: Maximum travel duration between machines
        seed: Random seed

    Returns:
        JSSPTransportInstance
    """
    if seed is not None:
        random.seed(seed)

    jssp_instance = generate_random_jssp_instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        min_duration=min_op_duration,
        max_duration=max_op_duration,
        seed=seed,
    )

    travel_times = [
        [
            random.randint(min_travel_duration, max_travel_duration)
            for _ in range(num_machines + 2)
        ]
        for _ in range(num_machines + 2)
    ]
    # ensure self-travel times are zero (for machines and any buffer nodes)
    for i in range(len(travel_times)):
        travel_times[i][i] = 0

    travel_times = [tuple(row) for row in travel_times]
    out_buf_idx = num_machines + 1
    duration = 0
    for job in jssp_instance:
        job.append((out_buf_idx, duration))
    return JSSPTransportInstance([jssp_instance, travel_times])


def generate_truncated_normal_transport_instance(
    num_jobs: int,
    num_machines: int,
    min_op_duration: int = 1,
    max_op_duration: int = 100,
    min_transport_duration: int = 1,
    max_transport_duration: int = 100,
    interval: int = 10,
    std: float = 5.0,
) -> JSSPTransportInstance:
    """
    Generate a random JSSP transport instance using truncated normal distribution for durations.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        max_operation_duration: Maximum duration of an operation
        max_travel_time: Maximum travel time between machines
    """

    jssp_instance = generate_truncated_normal_jssp_instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        min_duration=min_op_duration,
        max_duration=max_op_duration,
        interval=interval,
        std=std,
    )
    dist = _get_truncated_dist(
        max_duration=max_transport_duration,
        min_duration=min_transport_duration,
        interval=interval,
        std=std,
    )

    # travel_times = [
    #     [int(dist.rvs(1)) for _ in range(num_machines + 2)]
    #     for _ in range(num_machines + 2)
    # ]
    # print(f"Generated travel times (list):\n{travel_times}")
    travel_times = dist.rvs(size=(num_machines + 2, num_machines + 2)).astype(int)
    # print(f"Generated travel times:\n{travel_times}")
    # ensure self-travel times are zero (for machines and any buffer nodes)
    for i in range(len(travel_times)):
        travel_times[i][i] = 0

    travel_times = [tuple(row) for row in travel_times]
    out_buf_idx = num_machines + 1
    duration = 0
    for job in jssp_instance:
        job.append((out_buf_idx, duration))
    return JSSPTransportInstance([jssp_instance, travel_times])


def generate_norm_truncated_normal_transport_instance(
    num_jobs: int,
    num_machines: int,
    min_op_duration: int = 1,
    max_op_duration: int = 100,
    min_transport_duration: int = 1,
    max_transport_duration: int = 100,
    interval: int = 10,
    std: float = 5.0,
) -> JSSPTransportInstance:
    """
    Generate a random JSSP transport instance using truncated normal distribution for durations.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        max_operation_duration: Maximum duration of an operation
        max_travel_time: Maximum travel time between machines
    """

    jssp_instance = generate_norm_truncated_normal_jssp_instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        min_duration=min_op_duration,
        max_duration=max_op_duration,
        interval=interval,
        std=std,
    )
    dist = _get_norm_truncated_dist(
        max_duration=max_transport_duration,
        min_duration=min_transport_duration,
        interval=interval,
        std=std,
    )

    # travel_times = [
    #     [int(dist.rvs(1)) for _ in range(num_machines + 2)]
    #     for _ in range(num_machines + 2)
    # ]
    # print(f"Generated travel times (list):\n{travel_times}")
    travel_times = dist.rvs(size=(num_machines + 2, num_machines + 2)).astype(int)
    # print(f"Generated travel times:\n{travel_times}")
    # ensure self-travel times are zero (for machines and any buffer nodes)
    for i in range(len(travel_times)):
        travel_times[i][i] = 0

    travel_times = [tuple(row) for row in travel_times]
    out_buf_idx = num_machines + 1
    duration = 0
    for job in jssp_instance:
        job.append((out_buf_idx, duration))
    return JSSPTransportInstance([jssp_instance, travel_times])


def generate_ratio_transport_instance(
    num_jobs: int,
    num_machines: int,
    min_operation_duration: int = 1,
    max_operation_duration: int = 10,
    min_travel_time: int = 1,
    max_travel_time: int = 5,
    jssp_to_agv_ratio: float = 1.0,
) -> JSSPTransportInstance:
    """
    Generate a random JSSP transport instance with scaled durations based on a JSSP-to-AGV ratio.

    The ratio controls the relative importance of JSSP operations vs AGV transport:
    - ratio > 1.0: JSSP operations dominate (longer operation times relative to travel)
    - ratio = 1.0: Balanced (no scaling applied)
    - ratio < 1.0: AGV transport dominates (longer travel times relative to operations)

    The scaling is applied as follows:
    - Operation durations are multiplied by sqrt(ratio)
    - Travel times are divided by sqrt(ratio)

    This ensures that:
    - ratio = 4.0 means operations are 2x longer, travel times are 2x shorter
    - ratio = 0.25 means operations are 2x shorter, travel times are 2x longer

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        min_operation_duration: Minimum duration of an operation (before scaling)
        max_operation_duration: Maximum duration of an operation (before scaling)
        min_travel_time: Minimum travel time between machines (before scaling)
        max_travel_time: Maximum travel time between machines (before scaling)
        jssp_to_agv_ratio: Ratio of JSSP operation importance to AGV transport importance.
                          Values > 1 increase operation times relative to travel times.
                          Values < 1 increase travel times relative to operation times.

    Returns:
        JSSPTransportInstance with scaled durations
    """
    import math

    if jssp_to_agv_ratio <= 0:
        raise ValueError("jssp_to_agv_ratio must be positive")

    # Calculate scaling factors using square root for balanced scaling
    operation_scale = math.sqrt(jssp_to_agv_ratio)
    travel_scale = 1.0 / math.sqrt(jssp_to_agv_ratio)

    # Scale the duration bounds
    scaled_min_op_duration = max(1, int(min_operation_duration * operation_scale))
    scaled_max_op_duration = max(1, int(max_operation_duration * operation_scale))

    # Scale the travel time bounds
    scaled_min_travel = max(1, int(min_travel_time * travel_scale))
    scaled_max_travel = max(1, int(max_travel_time * travel_scale))

    # Ensure min <= max after scaling
    if scaled_min_op_duration > scaled_max_op_duration:
        scaled_min_op_duration = scaled_max_op_duration
    if scaled_min_travel > scaled_max_travel:
        scaled_min_travel = scaled_max_travel

    # Generate JSSP instance with scaled operation durations
    jssp_instance = generate_random_jssp_instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        min_duration=scaled_min_op_duration,
        max_duration=scaled_max_op_duration,
    )

    # Generate travel times with scaled bounds
    travel_times = [
        [
            random.randint(scaled_min_travel, scaled_max_travel)
            for _ in range(num_machines + 2)
        ]
        for _ in range(num_machines + 2)
    ]

    # Ensure self-travel times are zero
    for i in range(len(travel_times)):
        travel_times[i][i] = 0

    # Convert each row to an immutable tuple
    travel_times = [tuple(row) for row in travel_times]

    # Add output buffer operation to each job
    out_buf_idx = num_machines + 1
    duration = 0
    for job in jssp_instance:
        job.append((out_buf_idx, duration))

    return JSSPTransportInstance([jssp_instance, travel_times])


def generate_grid_uniform_transport_instance(
    num_jobs: int,
    num_machines: int,
    min_op_duration: int = 1,
    max_op_duration: int = 100,
    min_transport_duration: int = 1,
    max_transport_duration: int = 100,
) -> JSSPTransportInstance:
    """
    Generate a grid-based JSSP transport instance.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        min_op_duration: Minimum duration of an operation
        max_op_duration: Maximum duration of an operation
        min_transport_duration: Minimum travel time between machines
        max_transport_duration: Maximum travel time between machines

    """
    jssp_instance = generate_random_jssp_instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        min_duration=min_op_duration,
        max_duration=max_op_duration,
    )

    travel_times = [
        [
            random.randint(min_transport_duration, max_transport_duration)
            for _ in range(num_machines + 2)
        ]
        for _ in range(num_machines + 2)
    ]
    # ensure self-travel times are zero (for machines and any buffer nodes)
    for i in range(len(travel_times)):
        travel_times[i][i] = 0

    travel_times = [tuple(row) for row in travel_times]
    out_buf_idx = num_machines + 1
    duration = 0
    for job in jssp_instance:
        job.append((out_buf_idx, duration))
    return JSSPTransportInstance([jssp_instance, travel_times])


def generate_grid_skewed_transport_instance(
    num_jobs: int,
    num_machines: int,
    min_op_duration: int = 1,
    max_op_duration: int = 100,
    min_transport_duration: int = 1,
    max_transport_duration: int = 100,
) -> JSSPTransportInstance:
    """
    Generate a grid-based JSSP transport instance.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        min_op_duration: Minimum duration of an operation
        max_op_duration: Maximum duration of an operation
        min_transport_duration: Minimum travel time between machines
        max_transport_duration: Maximum travel time between machines

    """
    jssp_instance = generate_random_jssp_instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        min_duration=min_op_duration,
        max_duration=max_op_duration,
    )

    travel_times = [
        [
            random.randint(min_transport_duration, max_transport_duration)
            for _ in range(num_machines + 2)
        ]
        for _ in range(num_machines + 2)
    ]
    # ensure self-travel times are zero (for machines and any buffer nodes)
    for i in range(len(travel_times)):
        travel_times[i][i] = 0

    travel_times = [tuple(row) for row in travel_times]
    out_buf_idx = num_machines + 1
    duration = 0
    for job in jssp_instance:
        job.append((out_buf_idx, duration))
    return JSSPTransportInstance([jssp_instance, travel_times])


def get_transport_instance_info(
    instance: JSSPTransportInstance,
) -> str:
    """
    Get a string representation of the JSSP transport instance information.

    Args:
        instance: JSSPTransportInstance

    Returns:
        String representation of the instance information
    """
    jssp_info = get_instance_info(instance[0])
    print()
    print("================================")
    print(f"Getting jssp_info: {jssp_info}")
    travel_time_info = f"Travel Times: {instance[1]}"
    return f"{jssp_info}\n{travel_time_info}"


if __name__ == "__main__":
    # Test the module
    print("Testing transport instance loading...")
    _load_transport_instance()

    # Test random generation
    print("\n=== Testing generate_random_transport_instance ===")
    random_instance = generate_random_transport_instance(3, 4)
    print(f"Random instance info: {get_transport_instance_info(random_instance)}")

    # Test ratio-based generation
    print("\n=== Testing generate_ratio_transport_instance ===")

    # Balanced ratio (same as random)
    print("\n--- Ratio 1.0 (balanced) ---")
    balanced_instance = generate_ratio_transport_instance(3, 4, jssp_to_agv_ratio=1.0)
    print(f"Balanced instance info: {get_transport_instance_info(balanced_instance)}")

    # JSSP-dominated (longer operations, shorter travel)
    print("\n--- Ratio 4.0 (JSSP-dominated) ---")
    jssp_dominated = generate_ratio_transport_instance(3, 4, jssp_to_agv_ratio=4.0)
    print(
        f"JSSP-dominated instance info: {get_transport_instance_info(jssp_dominated)}"
    )

    # AGV-dominated (shorter operations, longer travel)
    print("\n--- Ratio 0.25 (AGV-dominated) ---")
    agv_dominated = generate_ratio_transport_instance(3, 4, jssp_to_agv_ratio=0.25)
    print(f"AGV-dominated instance info: {get_transport_instance_info(agv_dominated)}")
