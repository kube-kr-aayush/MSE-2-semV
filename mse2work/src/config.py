from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
NASA_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + SENSOR_COLUMNS
FEATURE_COLUMNS = ["cycle", "setting_1", "setting_2", "setting_3"] + SENSOR_COLUMNS

FAILURE_THRESHOLD = 30
RANDOM_STATE = 42
