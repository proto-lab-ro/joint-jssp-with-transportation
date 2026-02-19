# Predefined parameter sets
import subprocess

from jssp_agv.benchmark.experiments_config import all_configs


for cfg in all_configs:
    cmd = (
        f"uv run src/jssp_agv/benchmark/modules/grid_comparison_runner.py "
        f"--num_jobs={cfg['num_jobs']} "
        f"--num_machines={cfg['num_machines']} "
        f"--num_agvs={cfg['num_agvs']} "
        f"--interval={cfg['interval']} "
        f"--std={cfg['std']} "
        f"--num_instances={cfg['num_instances']} "
    )
    print("Running:", cmd)
    subprocess.run(cmd, shell=True)
