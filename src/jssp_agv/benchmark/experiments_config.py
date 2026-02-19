std = 5.0
interval = 10
num_instances = 20


_15x10 = {
    "num_jobs": 15,
    "num_machines": 10,
    "interval": interval,
    "std": std,
    "num_instances": num_instances,
}

_15x15 = {
    "num_jobs": 15,
    "num_machines": 15,
    "interval": interval,
    "std": std,
    "num_instances": num_instances,
}


_12x12 = {
    "num_jobs": 12,
    "num_machines": 12,
    "interval": interval,
    "std": std,
    "num_instances": num_instances,
}

_14x14 = {
    "num_jobs": 14,
    "num_machines": 14,
    "interval": interval,
    "std": std,
    "num_instances": num_instances,
}

_20x5 = {
    "num_jobs": 20,
    "num_machines": 5,
    "interval": interval,
    "std": std,
    "num_instances": num_instances,
}

_5x10 = {
    "num_jobs": 5,
    "num_machines": 10,
    "interval": interval,
    "std": std,
    "num_instances": num_instances,
}

_30x10 = {
    "num_jobs": 30,
    "num_machines": 10,
    "interval": interval,
    "std": std,
    "num_instances": num_instances,
}
_10x10 = {
    "num_jobs": 10,
    "num_machines": 10,
    "interval": interval,
    "std": std,
    "num_instances": num_instances,
}


_15x10_configs = [{**_15x10, "num_agvs": nr_agv} for nr_agv in [18, 15, 12, 9, 6, 3]]
_15x15_configs = [{**_15x15, "num_agvs": nr_agv} for nr_agv in [18, 15, 12, 9, 6, 3]]

_12x12_configs = [{**_12x12, "num_agvs": nr_agv} for nr_agv in [14, 12, 10, 7, 5, 2]]

_14x14_configs = [{**_14x14, "num_agvs": nr_agv} for nr_agv in [17, 14, 11, 8, 6, 3]]
_20x5_configs = [{**_20x5, "num_agvs": nr_agv} for nr_agv in [24, 20, 16, 12, 8, 4]]
_30x10_configs = [{**_30x10, "num_agvs": nr_agv} for nr_agv in [36, 30, 24, 18, 12, 6]]
_10x10_configs = [{**_10x10, "num_agvs": nr_agv} for nr_agv in [12, 10, 8, 6, 4, 2]]
_5x10_configs = [{**_5x10, "num_agvs": nr_agv} for nr_agv in [6, 5, 4, 3, 2, 1]]


all_configs = (
    _15x10_configs
    + _15x15_configs
    + _12x12_configs
    + _14x14_configs
    + _20x5_configs
    + _30x10_configs
    + _10x10_configs
    + _5x10_configs
)
