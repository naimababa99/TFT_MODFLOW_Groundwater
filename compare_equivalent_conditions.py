"""
Equivalent-conditions model comparison for R3 Major 8.

Reviewer comment: MODFLOW is scored at monthly resolution over its full
calibration period, while TFT/Hybrid are scored at daily resolution on
val/test only -- so the numbers in the results table aren't directly
comparable. This script builds a single table where every model is scored
over the SAME test period at BOTH temporal resolutions:

  - Daily, test-period-only (2024-01-01 .. 2025-12-12): persistence,
    seasonal persistence, AR(1), AR(7) (from baseline_forecasts.py),
    TFT-only, Hybrid, and MODFLOW linearly interpolated to daily (labeled
    as such -- it's interpolation skill between 24 monthly points, not a
    genuine daily simulation).
  - Monthly, test-period-only: TFT-only and Hybrid daily predictions
    resampled to month-start means, next to MODFLOW's own native monthly
    test-split rows (already computed by _run_modflow_vf.py).

It also runs a moving-block bootstrap on the paired daily test errors of
Hybrid vs. TFT-only to test whether the hybrid model's improvement over
the standalone TFT is statistically significant, as the reviewer requested.

Prerequisite: `_run_hybrid_tft.py` (main run, MODFLOW features on) and
`run_ablation_no_modflow.py` (MODFLOW features off) must both have been run
with the corrected evaluation harness first, producing
`models/hybrid_tft/hybrid_predictions.csv` and
`models/ablation_no_modflow/hybrid_predictions.csv` respectively.
"""

import os
import numpy as np
import pandas as pd

from baseline_forecasts import load_full_series, calculate_metrics


TEST_START = "2024-01-01"


def build_time_idx_lookup(processed_dir="data/processed"):
    """
    Same concat/sort/time_idx construction _run_hybrid_tft.py uses, so
    time_idx values in the saved prediction CSVs resolve to the correct
    calendar dates here.
    """
    full_df, scalers_dict = load_full_series(processed_dir)
    full_df["time_idx"] = np.arange(len(full_df))
    return full_df, scalers_dict


def load_model_daily_test_predictions(predictions_csv, full_df, target_scaler, target_col="water_level"):
    """
    Load a hybrid_predictions.csv (columns: time_idx, observed, predicted,
    residual -- scaled units), resolve dates via time_idx, inverse-transform
    to feet, and return a DataFrame with columns [date, observed_ft, predicted_ft]
    restricted to the true test split.
    """
    preds = pd.read_csv(predictions_csv)
    merged = preds.merge(full_df[["time_idx", "date", "split"]], on="time_idx", how="left")

    n_not_test = (merged["split"] != "test").sum()
    if n_not_test:
        print(f"    NOTE: dropping {n_not_test} rows outside the true test split from {predictions_csv}")
    merged = merged[merged["split"] == "test"].copy()

    merged["observed_ft"] = target_scaler.inverse_transform(merged[["observed"]].values).flatten()
    merged["predicted_ft"] = target_scaler.inverse_transform(merged[["predicted"]].values).flatten()

    return merged[["date", "observed_ft", "predicted_ft"]].sort_values("date").reset_index(drop=True)


def interpolate_modflow_daily(modflow_csv, date_col="date", value_col="simulated_depth_ft"):
    """
    Reproduces _run_hybrid_tft.py's _interpolate_modflow_to_daily: linear
    interpolation of MODFLOW's monthly output onto a daily grid, ffill/bfill
    at the edges. Returns a daily Series indexed by date.
    """
    modflow_df = pd.read_csv(modflow_csv, parse_dates=[date_col])
    daily_dates = pd.date_range(start=modflow_df[date_col].min(), end=modflow_df[date_col].max(), freq="D")
    daily_df = pd.DataFrame({date_col: daily_dates}).merge(
        modflow_df[[date_col, value_col]], on=date_col, how="left"
    )
    daily_df[value_col] = daily_df[value_col].interpolate(method="linear", limit_direction="both")
    daily_df[value_col] = daily_df[value_col].ffill().bfill()
    return daily_df.set_index(date_col)[value_col]


def moving_block_bootstrap_rmse_diff(errors_a, errors_b, block_size=30, n_boot=2000, seed=42):
    """
    Paired moving-block bootstrap on daily forecast errors of two models
    (errors_a, errors_b -- same dates, same order). Returns the bootstrap
    distribution of RMSE(a) - RMSE(b), a 95% CI, and a two-sided p-value
    for the null that the true difference is zero.

    Block bootstrap (rather than i.i.d. resampling of individual days) is
    used because daily groundwater forecast errors are autocorrelated --
    resampling single days independently would understate the true
    uncertainty in the RMSE estimates.
    """
    rng = np.random.default_rng(seed)
    n = len(errors_a)
    assert len(errors_b) == n
    n_blocks = int(np.ceil(n / block_size))

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        rmse_a = np.sqrt(np.mean(errors_a[idx] ** 2))
        rmse_b = np.sqrt(np.mean(errors_b[idx] ** 2))
        diffs[b] = rmse_a - rmse_b

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: fraction of resamples on the other side of zero from the mean, doubled
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = min(p_value, 1.0)

    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant_at_0.05": bool(ci_low > 0 or ci_high < 0),
        "p_value_approx": float(p_value),
    }


def main():
    print("=" * 70)
    print("   EQUIVALENT-CONDITIONS MODEL COMPARISON (R3 Major 8)")
    print("=" * 70)

    full_df, scalers_dict = build_time_idx_lookup()
    target_scaler = scalers_dict["target_scaler"]

    test_daily_observed = full_df[full_df["split"] == "test"][["date", "water_level_original"]].rename(
        columns={"water_level_original": "observed_ft"}
    )

    rows = []

    # --- Baselines (already daily, test-only, in results/baseline_metrics.csv) ---
    baseline_df = pd.read_csv("results/baseline_metrics.csv")
    baseline_test = baseline_df[baseline_df["split"] == "test"]
    for _, r in baseline_test.iterrows():
        rows.append({
            "model": r["model"], "resolution": "daily", "period": "test",
            "n": r["n"], "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"],
        })

    # --- TFT-only and Hybrid: daily test + monthly-aggregated test ---
    model_daily = {}
    for label, csv_path in [
        ("TFT-only (no MODFLOW)", "models/ablation_no_modflow/hybrid_predictions.csv"),
        ("Hybrid TFT-MODFLOW", "models/hybrid_tft/hybrid_predictions.csv"),
    ]:
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping {label}")
            continue

        print(f"\n  Loading {label} from {csv_path}...")
        daily = load_model_daily_test_predictions(csv_path, full_df, target_scaler)
        model_daily[label] = daily

        m_daily = calculate_metrics(daily["observed_ft"], daily["predicted_ft"])
        rows.append({
            "model": label, "resolution": "daily", "period": "test",
            "n": m_daily["n"], "RMSE": m_daily["RMSE"], "MAE": m_daily["MAE"], "R2": m_daily["R2"],
        })

        monthly = daily.set_index("date").resample("MS").mean()
        m_monthly = calculate_metrics(monthly["observed_ft"], monthly["predicted_ft"])
        rows.append({
            "model": label, "resolution": "monthly (daily preds resampled)", "period": "test",
            "n": m_monthly["n"], "RMSE": m_monthly["RMSE"], "MAE": m_monthly["MAE"], "R2": m_monthly["R2"],
        })

    # --- MODFLOW: native monthly test rows ---
    modflow_csv = "models/modflow_enhanced/enhanced_predictions.csv"
    modflow_df = pd.read_csv(modflow_csv, parse_dates=["date"])
    modflow_test = modflow_df[modflow_df["split"] == "test"]
    m_modflow_monthly = calculate_metrics(modflow_test["observed_depth_ft"], modflow_test["simulated_depth_ft"])
    rows.append({
        "model": "MODFLOW (native monthly)", "resolution": "monthly", "period": "test",
        "n": m_modflow_monthly["n"], "RMSE": m_modflow_monthly["RMSE"],
        "MAE": m_modflow_monthly["MAE"], "R2": m_modflow_monthly["R2"],
    })

    # --- MODFLOW: linearly interpolated to daily, test period only ---
    modflow_daily = interpolate_modflow_daily(modflow_csv)
    modflow_daily_test = test_daily_observed.merge(
        modflow_daily.rename("predicted_ft"), left_on="date", right_index=True, how="inner"
    )
    m_modflow_daily = calculate_metrics(modflow_daily_test["observed_ft"], modflow_daily_test["predicted_ft"])
    rows.append({
        "model": "MODFLOW (monthly, linearly interpolated to daily)", "resolution": "daily", "period": "test",
        "n": m_modflow_daily["n"], "RMSE": m_modflow_daily["RMSE"],
        "MAE": m_modflow_daily["MAE"], "R2": m_modflow_daily["R2"],
    })

    results_df = pd.DataFrame(rows)[["model", "resolution", "period", "n", "RMSE", "MAE", "R2"]]
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/equivalent_conditions_comparison.csv", index=False)

    print("\n" + "-" * 70)
    print("  EQUIVALENT-CONDITIONS COMPARISON (feet, original units)")
    print("-" * 70)
    print(results_df.to_string(index=False))
    print(f"\n  Saved to: results/equivalent_conditions_comparison.csv")

    # --- Bootstrap significance: Hybrid vs. TFT-only, daily test errors ---
    if "TFT-only (no MODFLOW)" in model_daily and "Hybrid TFT-MODFLOW" in model_daily:
        tft = model_daily["TFT-only (no MODFLOW)"]
        hyb = model_daily["Hybrid TFT-MODFLOW"]
        merged = tft.merge(hyb, on="date", suffixes=("_tft", "_hybrid"))
        assert (merged["observed_ft_tft"].to_numpy() == merged["observed_ft_hybrid"].to_numpy()).all(), (
            "TFT-only and Hybrid test sets must cover identical dates for a paired comparison"
        )

        errors_tft = (merged["observed_ft_tft"] - merged["predicted_ft_tft"]).to_numpy()
        errors_hybrid = (merged["observed_ft_hybrid"] - merged["predicted_ft_hybrid"]).to_numpy()

        boot = moving_block_bootstrap_rmse_diff(errors_tft, errors_hybrid, block_size=30, n_boot=2000)
        boot_df = pd.DataFrame([{
            "comparison": "RMSE(TFT-only) - RMSE(Hybrid)",
            "n_days": len(merged),
            **boot,
        }])
        boot_df.to_csv("results/bootstrap_significance.csv", index=False)

        print("\n" + "-" * 70)
        print("  BOOTSTRAP SIGNIFICANCE: Hybrid vs. TFT-only (daily test RMSE)")
        print("-" * 70)
        print(boot_df.to_string(index=False))
        print(f"\n  Saved to: results/bootstrap_significance.csv")
    else:
        print("\n  Skipping bootstrap significance test -- need both TFT-only and Hybrid predictions.")


if __name__ == "__main__":
    main()
