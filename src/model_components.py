"""Fold-fitted numeric transformations and class-weight helper from the notebook."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NumericStrategy(BaseEstimator, TransformerMixin):
    """Fold-fitted raw, percentile-clipping, or signed-log numeric transform."""
    def __init__(self, strategy="S0_Raw", zero_threshold=0.10):
        self.strategy = strategy
        self.zero_threshold = zero_threshold

    def _array(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.nan
        return arr

    def fit(self, X, y=None):
        arr = self._array(X)
        self.n_features_in_ = arr.shape[1]
        self.medians_ = np.zeros(self.n_features_in_, dtype=float)
        for j in range(self.n_features_in_):
            valid = arr[:, j][np.isfinite(arr[:, j])]
            self.medians_[j] = float(np.median(valid)) if len(valid) else 0.0
        filled = np.where(np.isfinite(arr), arr, self.medians_[None, :])
        self.lower_ = np.percentile(filled, 1, axis=0)
        self.upper_ = np.percentile(filled, 99, axis=0)
        self.zero_columns_ = np.mean(filled == 0, axis=0) >= self.zero_threshold
        return self

    def transform(self, X):
        arr = self._array(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("Numeric feature count changed after fit.")
        arr = np.where(np.isfinite(arr), arr, self.medians_[None, :])
        if self.strategy == "S0_Raw":
            return arr
        if self.strategy == "S1_Clipping":
            return np.clip(arr, self.lower_[None, :], self.upper_[None, :])
        if self.strategy == "S2_SignedLog":
            transformed = np.sign(arr) * np.log1p(np.abs(arr))
            if np.any(self.zero_columns_):
                indicators = (arr[:, self.zero_columns_] == 0).astype(float)
                transformed = np.hstack([transformed, indicators])
            return transformed
        raise ValueError(f"Unknown strategy: {self.strategy}")

def compute_mild_class_weights(y, exponent=0.35):
    """Fractional inverse-frequency weights normalized to mean one."""
    y_series = pd.Series(np.asarray(y).astype(str))
    counts = y_series.value_counts()
    inverse_frequency = len(y_series) / (len(counts) * counts)
    class_weights = inverse_frequency.astype(float) ** float(exponent)
    weights = y_series.map(class_weights).to_numpy(dtype=float)
    weights /= max(float(weights.mean()), 1e-12)
    return weights
