import pandas as pd
from .config import DATA_RAW, NASA_COLUMNS, FAILURE_THRESHOLD, FEATURE_COLUMNS

REQUIRED_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

def _read_nasa_file(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=NASA_COLUMNS)

def load_training_frame():
    path = DATA_RAW / "train_FD001.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Copy the official NASA FD001 files into data/raw/.")
    df = _read_nasa_file(path)
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["rul"] = df["engine_id"].map(max_cycles) - df["cycle"]
    df["failure_soon"] = (df["rul"] <= FAILURE_THRESHOLD).astype(int)
    return df

def load_incoming_frame():
    test_path = DATA_RAW / "test_FD001.txt"
    rul_path = DATA_RAW / "RUL_FD001.txt"
    if not test_path.exists() or not rul_path.exists():
        raise FileNotFoundError("Missing test_FD001.txt or RUL_FD001.txt in data/raw/.")
    df = _read_nasa_file(test_path)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    engines = sorted(df["engine_id"].unique())
    if len(engines) != len(rul):
        raise ValueError("RUL_FD001.txt row count does not match test engine count.")
    final_rul = dict(zip(engines, rul))
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["rul"] = (
        df["engine_id"].map(max_cycles) - df["cycle"]
        + df["engine_id"].map(final_rul)
    )
    df["failure_soon"] = (df["rul"] <= FAILURE_THRESHOLD).astype(int)
    return df

def feature_frame(df):
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLUMNS].astype(float)
