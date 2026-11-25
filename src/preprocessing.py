"""Data validation and deterministic feature engineering for UNSW-NB15."""
from __future__ import annotations
import numpy as np
import pandas as pd

EXPECTED_UNSW_CLASSES = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
]
EXCLUDED_PREDICTORS = ["id", "label", "attack_cat"]
KNOWN_CATEGORICAL = ["proto", "service", "state"]

def clean_column_names(frame: pd.DataFrame, partition_name: str = "input") -> pd.DataFrame:
    frame = frame.copy()
    cleaned = [str(c).strip() for c in frame.columns]
    duplicated = pd.Index(cleaned)[pd.Index(cleaned).duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{partition_name} contains duplicate column names after trimming: {duplicated}")
    frame.columns = cleaned
    return frame

def normalize_target_values(series: pd.Series, split_name: str = "input") -> pd.Series:
    canonical = {name.casefold(): name for name in EXPECTED_UNSW_CLASSES}
    cleaned = series.astype("string").str.strip()
    normalized = cleaned.str.casefold().map(canonical)
    unknown = sorted(cleaned[normalized.isna()].dropna().unique().tolist())
    if unknown:
        raise ValueError(f"Unknown {split_name} target labels: {unknown}")
    return normalized.astype(str)

def add_traffic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the notebook's seven deterministic traffic-derived predictors."""
    result = frame.copy()
    def numeric(name):
        return pd.to_numeric(result[name], errors="coerce")
    if {"sbytes", "dbytes"}.issubset(result.columns):
        src, dst = numeric("sbytes"), numeric("dbytes")
        result["derived_total_bytes"] = src + dst
        result["derived_src_dst_byte_log_ratio"] = np.log1p(src.clip(lower=0)) - np.log1p(dst.clip(lower=0))
    if {"spkts", "dpkts"}.issubset(result.columns):
        src, dst = numeric("spkts"), numeric("dpkts")
        result["derived_total_packets"] = src + dst
        result["derived_src_dst_packet_log_ratio"] = np.log1p(src.clip(lower=0)) - np.log1p(dst.clip(lower=0))
    if {"sbytes", "dbytes", "spkts", "dpkts"}.issubset(result.columns):
        total_bytes = numeric("sbytes") + numeric("dbytes")
        total_packets = numeric("spkts") + numeric("dpkts")
        result["derived_mean_bytes_per_packet"] = total_bytes / total_packets.clip(lower=1)
    if {"sbytes", "dbytes", "spkts", "dpkts", "dur"}.issubset(result.columns):
        total_bytes = numeric("sbytes") + numeric("dbytes")
        total_packets = numeric("spkts") + numeric("dpkts")
        safe_duration = numeric("dur").clip(lower=0) + 1e-6
        result["derived_bytes_per_duration"] = total_bytes / safe_duration
        result["derived_packets_per_duration"] = total_packets / safe_duration
    return result

def validate_partitions(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Validate required metadata and predictor schema; return raw predictor columns."""
    train_df = clean_column_names(train_df, "Training CSV")
    test_df = clean_column_names(test_df, "Testing CSV")
    required = {"id", "label", "attack_cat"}
    missing_train = sorted(required - set(train_df.columns))
    missing_test = sorted(required - set(test_df.columns))
    if missing_train or missing_test:
        raise ValueError(f"Required metadata missing. Training={missing_train}; testing={missing_test}")
    raw_features = [c for c in train_df.columns if c not in EXCLUDED_PREDICTORS]
    missing_test_features = sorted(set(raw_features) - set(test_df.columns))
    if missing_test_features:
        raise ValueError(f"Testing CSV is missing required predictors: {missing_test_features}")
    if train_df["attack_cat"].isna().any() or test_df["attack_cat"].isna().any():
        raise ValueError("Target values are missing; labelled records must not be silently dropped.")
    labels = normalize_target_values(train_df["attack_cat"], "training")
    if set(labels.unique()) != set(EXPECTED_UNSW_CLASSES):
        raise ValueError("Training data does not contain all expected ten UNSW-NB15 classes.")
    return train_df, test_df, raw_features, labels
