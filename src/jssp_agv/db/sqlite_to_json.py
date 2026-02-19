import json
import sqlite3
from pathlib import Path


DB_PATH = "paper_results.db"
OUTPUT_DIR = Path("json_export")

OUTPUT_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

for table in tables:
    cursor.execute(f"SELECT * FROM {table}")
    rows = cursor.fetchall()

    data = [dict(row) for row in rows]

    with open(OUTPUT_DIR / f"{table}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Exported {table}.json")

conn.close()
