GRID_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS grid_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    num_instances INTEGER NOT NULL,
    num_jobs INTEGER NOT NULL,
    num_machines INTEGER NOT NULL,
    num_agvs INTEGER NOT NULL,
    max_duration INTEGER,
    number_cells INTEGER,
    seed_start INTEGER,
    interval INTEGER,
    std REAL,
    grid_type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- timestamp for insertion
    UNIQUE(
        num_instances, num_jobs, num_machines, num_agvs,
        max_duration, number_cells, interval, std,
        grid_type, seed_start
    )
);

CREATE TABLE IF NOT EXISTS grid_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_experiments_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    agv_solver_type TEXT NOT NULL,
    job_solver_type TEXT NOT NULL,
    solver_type TEXT NOT NULL,
    solver_name TEXT NOT NULL,
    makespan REAL NOT NULL,
    job_model TEXT,
    agv_model TEXT,
    seed INTEGER,
    nr_inst INTEGER,
    min_travel INTEGER,
    min_op INTEGER,
    job_model_name TEXT, 
    agv_model_name TEXT,
    op2transport_ratio REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- timestamp for insertion
    FOREIGN KEY (grid_experiments_id)
        REFERENCES grid_experiments(id)
        ON DELETE CASCADE,
    UNIQUE(
        grid_experiments_id, instance_id, agv_solver_type,
        job_solver_type,solver_type,solver_name,
        makespan,job_model,agv_model,seed,nr_inst,min_travel,min_op
    )
);
"""
BM_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS bm_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    num_instances INTEGER NOT NULL,
    num_jobs INTEGER NOT NULL,
    num_machines INTEGER NOT NULL,
    num_agvs INTEGER NOT NULL,
    min_operation_duration INTEGER,
    max_operation_duration INTEGER,
    min_travel_time INTEGER,
    max_travel_time INTEGER,
    seed_start INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- timestamp for insertion
    UNIQUE(
        num_instances, num_jobs, num_machines, num_agvs,
        min_operation_duration, max_operation_duration,
        min_travel_time, max_travel_time, seed_start
    )
);

CREATE TABLE IF NOT EXISTS bm_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bm_experiments_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    agv_solver_type TEXT NOT NULL,
    job_solver_type TEXT NOT NULL,
    solver_type TEXT NOT NULL,
    solver_name TEXT NOT NULL,
    makespan REAL NOT NULL,
    job_model TEXT,
    agv_model TEXT,
    seed INTEGER,
    job_model_name TEXT, 
    agv_model_name TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- timestamp for insertion
    FOREIGN KEY (bm_experiments_id)
        REFERENCES bm_experiments(id)
        ON DELETE CASCADE,
    UNIQUE(
        bm_experiments_id, instance_id, agv_solver_type,
        job_solver_type,solver_type,solver_name,
        makespan,job_model,agv_model,seed
    )
);

"""

SOLVER_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS solvers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Predefined columns
    solver_name TEXT NOT NULL UNIQUE,        -- e.g., "07-24-44_M_LFalse_FTTrue/0"
    solver_parent TEXT NOT NULL,             -- e.g., "07-24-44_M_LFalse_FTTrue"
    instance_generator TEXT,                 -- from config: env.instance_generator
    reward_function TEXT,                    -- from config: env.reward_function
    job_observation_provider TEXT,           -- from config: env.job_observation_provider
    agv_observation_provider TEXT,           -- from config: env.agv_observation_provider
    total_frames INTEGER,                    -- from config: training.total_frames
    finetune BOOLEAN DEFAULT 0,              -- from config: finetune.finetune
    lora BOOLEAN DEFAULT 0,                  -- from config: finetune.lora

    -- Additional useful columns
    env_type TEXT,                           -- env type: e.g., env_type: aec
    number_agvs TEXT,                        -- can be integer or 'random'
    max_episode_steps INTEGER,               -- env.max_episode_steps
    seed INTEGER,                             -- global seed
    gnn BOOLEAN DEFAULT 0,                    -- finetune.gnn
    policy_head BOOLEAN DEFAULT 0,            -- finetune.policy_head
    rank INTEGER,                             -- finetune.rank
    alpha REAL,                               -- finetune.alpha
    dropout REAL,                             -- finetune.dropout
    ml_path TEXT,                             -- finetune.ml_path (path to ML checkpoint)
    log_dir TEXT,                             -- logging dir
    save_dir TEXT,                            -- checkpoint dir
    config_params JSON NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- timestamp for insertion

    -- Constraints
    UNIQUE(solver_name)
);
"""
MODEL_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    model_type TEXT NOT NULL,        -- e.g., "decoupled" or "coupled"
    jssp_model_path TEXT NOT NULL,        
    agv_model_path TEXT NOT NULL,   
    jssp_model_solver_id INTEGER NOT NULL,
    agv_model_solver_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (jssp_model_solver_id)
        REFERENCES solvers(id)
        ON DELETE CASCADE,
    FOREIGN KEY (agv_model_solver_id)
        REFERENCES solvers(id)
        ON DELETE CASCADE,
    
    UNIQUE(model_type, jssp_model_path, agv_model_path)
);
"""
