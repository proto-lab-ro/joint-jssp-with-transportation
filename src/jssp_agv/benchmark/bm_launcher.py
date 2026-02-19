import subprocess

from jssp_agv.benchmark.experiments_config import all_configs


configs_w_agvs = [
    {k: v for k, v in cfg.items() if k not in ["interval", "std", "num_instances"]}
    for cfg in all_configs
]

for cfg in configs_w_agvs:
    cmd = (
        f"uv run src/jssp_agv/benchmark/modules/comparison_runner.py "
        f"--num_jobs={cfg['num_jobs']} "
        f"--num_machines={cfg['num_machines']} "
        f"--num_agvs={cfg['num_agvs']}"
    )
    print("Running:", cmd)
    subprocess.run(cmd, shell=True)
