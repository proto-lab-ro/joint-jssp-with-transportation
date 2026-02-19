from pathlib import Path

# Lists of model paths to compare.
# Each path should point to the hydra output directory of a trained model.
from jssp_core.solver.heuristics import get_all_job_agv_heuristic_combinations


GNN_MODEL_PATHS = [
    Path("model_repo") / "01-23_07-52-10_M_LFalse_FTFalse",  #
]

LORA_MODEL_PATHS = []

AGV_MODEL_PATHS = [
    Path("model_repo") / "01-23_17-25-04_M_agv",  #  mor
    Path("model_repo") / "01-25_20-27-15_M_agv",  #  fddmwr
    Path("model_repo") / "01-27_15-27-34_M_agv",  #  lwr
    Path("model_repo") / "01-27_19-08-18_M_agv",  #  spt
    Path("model_repo") / "01-28_00-00-24_M_agv",  #  mwr
    Path("model_repo") / "01-29_07-45-23_M_agv",  #  random
    Path("model_repo") / "02-10_20-24-20_M_agv",  #  FCFS
    Path("model_repo") / "02-11_09-27-16_M_agv",  #  SMPT
    Path("model_repo") / "02-11_15-23-55_M_agv",  #  LPT
    Path("model_repo") / "02-11_20-34-26_M_agv",  #   LOR
]

JSSP_MODEL_PATHS = [
    Path("model_repo") / "01-23_10-47-36_M_jssp",  #  scpt
    Path("model_repo") / "01-25_13-50-00_M_jssp",  #  scta
    Path("model_repo") / "01-29_07-37-35_M_jssp",  #  random
    Path("model_repo") / "01-29_17-04-34_M_jssp",  #  sput
]

HEURISTIC_SOLVERS = get_all_job_agv_heuristic_combinations()
