import json
from pathlib import Path

import pandas as pd


def load_json_exports(json_dir: str | Path = "../../json_export"):
    """
    Loads all JSON files from the specified directory into pandas DataFrames.
    Returns a dictionary mapping the filename (without extension) to its DataFrame.
    """
    json_path = Path(json_dir).resolve()
    if not json_path.exists():
        # Try relative to the current file if the default doesn't work
        json_path = Path(__file__).parent.parent.parent / "json_export"
        json_path = json_path.resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"Could not find json_export directory at {json_path}")

    dataframes = {}
    for json_file in json_path.glob("*.json"):
        try:
            # Using orient='records' or default based on file structure
            # Most exported data from DBs is often in 'records' or default format
            # We'll try to load and if it's a list, it's usually records
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            df_name = f"df_{json_file.stem}"
            dataframes[df_name] = pd.DataFrame(data)
            print(f"Loaded {json_file.name} into {df_name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")

    return dataframes


if __name__ == "__main__":
    # Example usage
    dfs = load_json_exports()
    for name, df in dfs.items():
        print(f"\nDataFrame: {name}")
        print(df.head())
        print(f"Shape: {df.shape}")
