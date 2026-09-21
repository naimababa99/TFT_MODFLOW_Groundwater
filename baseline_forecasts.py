"""
Simple forecasting benchmarks for the 1-day-ahead groundwater-level task.

Added in response to a reviewer comment asking that the TFT/Hybrid model be
compared against persistence, seasonal persistence, and an autoregressive
model -- none of which existed anywhere in this repo before this file.

All three benchmarks are evaluated on the SAME train/val/test split used by
the rest of the pipeline (data_spliting_and_scalling.py), using only
information available strictly before the forecast origin:

  - Persistence:            y_hat[t] = y[t-1]
  - Seasonal persistence:   y_hat[t] = y[t-365]   (same calendar day, prior year)
  - Autoregressive AR(p):   y_hat[t] = c + sum_i phi_i * y[t-i], phi fit on
                             TRAIN ONLY, then applied using true past values
                             (never the model's own prior predictions) for
                             val/test -- the standard one-step-ahead AR
                             evaluation protocol, directly comparable to how
                             the TFT/Hybrid model is evaluated.

None of the three benchmarks are trained/fit on anything beyond the train
split; persistence and seasonal persistence have no fitted parameters at
all, so there is no leakage question for them beyond correct time indexing.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional


# =============================================================================
# 1. LOAD DATA (same processed splits used by the Hybrid model)
# =============================================================================
def load_full_series(
    processed_dir: str = "data/processed",
    date_col: str = "date",
    target_col: str = "water_level",
):
    """
    Load train/val/test scaled splits, concatenate into one continuous,
    date-sorted series, and inverse-transform the target back to its
    original units (feet) using the scalers fit during
    data_spliting_and_scalling.py (train-split statistics only).
    """
    train_df = pd.read_csv(os.path.join(processed_dir, "train_scaled.csv"), parse_dates=[date_col])
    val_df = pd.read_csv(os.path.join(processed_dir, "val_scaled.csv"), parse_dates=[date_col])
    test_df = pd.read_csv(os.path.join(processed_dir, "test_scaled.csv"), parse_dates=[date_col])

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(date_col).reset_index(drop=True)

    scalers_path = os.path.join(processed_dir, "scalers.joblib")
    scalers_dict = joblib.load(scalers_path)
    target_scaler = scalers_dict.get("target_scaler")

    if target_scaler is not None:
        full_df[f"{target_col}_original"] = target_scaler.inverse_transform(
            full_df[[target_col]]
        ).flatten()
    else:
        full_df[f"{target_col}_original"] = full_df[target_col]

    return full_df, scalers_dict


# =============================================================================
# 2. BENCHMARK FORECASTS
# =============================================================================
def persistence_forecast(series: pd.Series, horizon: int = 1) -> pd.Series:
    """y_hat[t] = y[t-horizon]. No fitted parameters."""
    return series.shift(horizon)


def seasonal_persistence_forecast(series: pd.Series, season_lag: int = 365) -> pd.Series:
    """y_hat[t] = y[t-season_lag] (same calendar day, prior year by default)."""
    return series.shift(season_lag)


def fit_ar_model(train_series: np.ndarray, lag_order: int = 1) -> np.ndarray:
    """
    Fit y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} by ordinary least
    squares on the TRAIN series only. Returns coefficients
    [c, phi_1, ..., phi_p].
    """
    n = len(train_series)
    if n <= lag_order:
        raise ValueError(f"Need more than {lag_order} train observations to fit AR({lag_order}).")

    X = np.ones((n - lag_order, lag_order + 1))
    for i in range(1, lag_order + 1):
        X[:, i] = train_series[lag_order - i: n - i]
    y = train_series[lag_order:]

    # Drop rows with NaNs (from missing raw readings, not just lag warm-up)
    # before fitting -- np.linalg.lstsq fails to converge if NaNs are present.
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def ar_forecast(full_series: np.ndarray, coeffs: np.ndarray, lag_order: int) -> np.ndarray:
    """
    One-step-ahead AR(p) forecast at every index t using the TRUE past
    values y[t-1..t-p] from full_series (never the model's own previous
    predictions) -- i.e. the standard walk-forward evaluation protocol.
    coeffs must come from fit_ar_model() on the train split only.
    """
    n = len(full_series)
    preds = np.full(n, np.nan)
    c = coeffs[0]
    phis = coeffs[1:]
    for t in range(lag_order, n):
        lags = full_series[t - lag_order: t][::-1]  # [y[t-1], y[t-2], ..., y[t-p]]
        preds[t] = c + np.dot(phis, lags)
    return preds


# =============================================================================
# 3. METRICS
# =============================================================================
def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)

    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]

    if len(obs) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "NSE": np.nan, "n": 0}

    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    mae = float(np.mean(np.abs(pred - obs)))

    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NSE": r2, "n": int(len(obs))}


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
def run_baseline_benchmarks(
    processed_dir: str = "data/processed",
    output_path: str = "results/baseline_metrics.csv",
    date_col: str = "date",
    target_col: str = "water_level",
    ar_orders: Optional[List[int]] = None,
    seasonal_lag: int = 365,
) -> pd.DataFrame:
    """
    Fit/evaluate persistence, seasonal persistence, and AR(p) benchmarks
    on the same train/val/test split as the TFT/Hybrid model, and save a
    results table comparable to the metrics reported for MODFLOW/Hybrid.
    """
    if ar_orders is None:
        ar_orders = [1, 7]

    print("=" * 70)
    print("     BASELINE FORECAST BENCHMARKS (persistence / seasonal / AR)")
    print("=" * 70)

    full_df, _ = load_full_series(processed_dir, date_col, target_col)
    y_col = f"{target_col}_original"
    y_full = full_df[y_col].to_numpy()
    train_mask = (full_df["split"] == "train").to_numpy()
    val_mask = (full_df["split"] == "val").to_numpy()
    test_mask = (full_df["split"] == "test").to_numpy()

    print(f"\n  Total records: {len(full_df)}")
    print(f"  Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    results_rows = []

    # --- Persistence (y_hat[t] = y[t-1]) ---
    pred_persistence = persistence_forecast(full_df[y_col], horizon=1).to_numpy()
    for split_name, mask in [("val", val_mask), ("test", test_mask)]:
        m = calculate_metrics(y_full[mask], pred_persistence[mask])
        m.update({"model": "persistence", "split": split_name})
        results_rows.append(m)

    # --- Seasonal persistence (y_hat[t] = y[t-365]) ---
    pred_seasonal = seasonal_persistence_forecast(full_df[y_col], season_lag=seasonal_lag).to_numpy()
    for split_name, mask in [("val", val_mask), ("test", test_mask)]:
        m = calculate_metrics(y_full[mask], pred_seasonal[mask])
        m.update({"model": f"seasonal_persistence_lag{seasonal_lag}", "split": split_name})
        results_rows.append(m)

    # --- AR(p), fit on TRAIN only, walk-forward one-step-ahead on val/test ---
    y_train = y_full[train_mask]
    for p in ar_orders:
        coeffs = fit_ar_model(y_train, lag_order=p)
        pred_ar = ar_forecast(y_full, coeffs, lag_order=p)
        for split_name, mask in [("val", val_mask), ("test", test_mask)]:
            m = calculate_metrics(y_full[mask], pred_ar[mask])
            m.update({"model": f"AR({p})", "split": split_name})
            results_rows.append(m)
        print(f"\n  AR({p}) coefficients (fit on train only):")
        print(f"    intercept: {coeffs[0]:.4f}")
        for i, phi in enumerate(coeffs[1:], start=1):
            print(f"    phi_{i}: {phi:.4f}")

    results_df = pd.DataFrame(results_rows)[["model", "split", "n", "RMSE", "MAE", "R2", "NSE"]]

    print("\n" + "-" * 70)
    print("  RESULTS (feet, original units)")
    print("-" * 70)
    print(results_df.to_string(index=False))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")

    return results_df


# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    run_baseline_benchmarks(
        processed_dir="data/processed",
        output_path="results/baseline_metrics.csv",
        date_col="date",
        target_col="water_level",
        ar_orders=[1, 7],
        seasonal_lag=365,
    )
