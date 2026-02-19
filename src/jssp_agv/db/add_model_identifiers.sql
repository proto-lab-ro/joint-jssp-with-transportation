-- SQLite
SELECT * FROM bm_results WHERE solver_type='ml';


ALTER TABLE bm_results
ADD COLUMN agv_model_identifier TEXT;

ALTER TABLE bm_results
ADD COLUMN job_model_identifier TEXT;

UPDATE bm_results
SET agv_model_identifier = 
    REPLACE(
        REPLACE(
            REPLACE(job_model, 'multirun\', ''),
            'multirun/', ''
        ),
        'model:repo\', ''
    ),
    'outputs\', ''
); 

UPDATE bm_results
SET job_model_identifier = 
    REPLACE(
        REPLACE(
            REPLACE(job_model, 'multirun\', ''),
            'multirun/', ''
        ),
        'model:repo\', ''
    ),
    'outputs\', ''
);


ALTER TABLE grid_results
ADD COLUMN agv_model_identifier TEXT;

ALTER TABLE grid_results
ADD COLUMN job_model_identifier TEXT;

UPDATE grid_results
SET agv_model_identifier = 
    REPLACE(
        REPLACE(
            REPLACE(job_model, 'multirun\', ''),
            'multirun/', ''
        ),
        'model:repo\', ''
    ),
    'outputs\', ''
); 

UPDATE grid_results
SET job_model_identifier = 
    REPLACE(
        REPLACE(
            REPLACE(job_model, 'multirun\', ''),
            'multirun/', ''
        ),
        'model:repo\', ''
    ),
    'outputs\', ''
); 