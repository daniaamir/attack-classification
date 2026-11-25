"""Prediction helpers for a saved sklearn-compatible NIDS pipeline."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from .preprocessing import add_traffic_features

def load_model(model_path):
    """Load a model artifact saved with joblib."""
    return joblib.load(model_path)

def prepare_prediction_frame(raw_frame, raw_feature_columns, numeric_columns):
    """Apply the same derived-feature logic and predictor ordering used at training."""
    frame = raw_frame.copy()
    missing = [c for c in raw_feature_columns if c not in frame.columns]
    if missing:
        raise ValueError("Missing required traffic features: " + ", ".join(missing))
    prepared = add_traffic_features(frame.reindex(columns=raw_feature_columns))
    for col in numeric_columns:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return prepared

def classify_csv(model, csv_path, raw_feature_columns, numeric_columns, labels):
    """Classify a CSV and return the enriched prediction dataframe."""
    original = pd.read_csv(csv_path, low_memory=False)
    model_input = prepare_prediction_frame(original, raw_feature_columns, numeric_columns)
    predictions = model.predict(model_input)
    probabilities = model.predict_proba(model_input)
    output = original.copy()
    output["predicted_attack_cat"] = predictions
    output["risk_level"] = ["Normal" if str(p).strip().lower() == "normal" else "Attack" for p in predictions]
    output["prediction_confidence"] = np.max(probabilities, axis=1)
    return output
