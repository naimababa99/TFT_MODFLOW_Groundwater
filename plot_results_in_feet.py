"""
Re-plot a saved TFT/hybrid run in ORIGINAL UNITS (feet, depth to water).

The training pipelines evaluate and plot in the scaled target space, so
`hybrid_results.png` carries "Water Level (scaled)" axes. Reviewers and readers
need feet. This script rebuilds the same four-panel figure from the saved
predictions CSV, inverse-transforming with the SAME scaler the pipeline fitted
on the train split (data/processed/scalers.joblib), and recomputes the metrics
in feet.

Nothing is re-run and no model is loaded: this reads
`<run_dir>/hybrid_predictions.csv` (columns time_idx, observed, predicted),
so it works for the hybrid run, the MODFLOW-prior variants and the
ablation_tests runs alike.

What changes between scaled and feet:
  - RMSE / MAE / bias are multiplied by the scaler's scale_ (they are in the
    target's units).
  - R2 / NSE are dimensionless and identical in both spaces.
  - The depth-to-water axis is inverted, so that a HIGHER water table (a
    smaller depth below the surface) appears higher in the plot.

Usage:
    python plot_results_in_feet.py                        # models/hybrid_tft
    python plot_results_in_feet.py <run_dir> [<run_dir> ...]
"""

import os
import sys
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROCESSED_DIR = "data/processed"
DATE_COL = "date"
TARGET_COL = "water_level"

# Figure title, panel titles, colors, line widths and grids below deliberately
# mirror plot_results() in the training pipelines
# (hybrid_tft_modflow_corrected_VF.ipynb / _run_hybrid_tft.py), so this figure
# is a drop-in replacement for hybrid_results.png and nothing but the units
# changes.
DEFAULT_TITLE = "Hybrid TFT-MODFLOW Groundwater Prediction"


# =============================================================================
# 1. DATA
# =============================================================================
def load_reference_series():
    """
    Rebuild the continuous daily series exactly as the pipelines build it
    (train+val+test concatenated, date-sorted, time_idx = 0..N-1) so that the
    saved predictions' time_idx values can be mapped back to dates, and load
    the train-fitted target scaler.
    """
    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{split}_scaled.csv"), parse_dates=[DATE_COL])
        df["split"] = split
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    full_df["time_idx"] = np.arange(len(full_df))

    scalers = joblib.load(os.path.join(PROCESSED_DIR, "scalers.joblib"))
    target_scaler = scalers.get("target_scaler")
    if target_scaler is None:
        raise SystemExit("No 'target_scaler' in scalers.joblib -- cannot convert to feet.")

    return full_df, target_scaler


def to_feet(values: np.ndarray, target_scaler) -> np.ndarray:
    return target_scaler.inverse_transform(np.asarray(values, dtype=float).reshape(-1, 1)).ravel()


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Same definitions as the pipelines' calculate_metrics, in feet."""
    mask = ~(np.isnan(observed) | np.isnan(predicted))
    obs, pred = observed[mask], predicted[mask]

    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    mae = float(np.mean(np.abs(pred - obs)))
    bias = float(np.mean(pred - obs))

    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"n": int(len(obs)), "RMSE": rmse, "MAE": mae, "Bias": bias, "R2": r2, "NSE": r2}


# =============================================================================
# 2. FIGURE
# =============================================================================
def plot_in_feet(dates, observed_ft, predicted_ft, metrics, title, save_path):
    """
    Four panels in the pipeline's own style (same colors, line widths, panel
    titles, grids and figure title as plot_results with is_scaled=False):
    time series, observed-vs-predicted scatter, residual series, residual
    histogram -- all in feet.
    """
    residuals = predicted_ft - observed_ft
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- time series ---------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(dates, observed_ft, 'b-', label='Observed', linewidth=2)
    ax1.plot(dates, predicted_ft, 'r--', label='Hybrid TFT-MODFLOW', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Depth to Water (feet)')
    ax1.set_title('Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # depth below surface: a deeper water table plots lower
    ax1.tick_params(axis='x', rotation=45)

    # --- scatter -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.scatter(observed_ft, predicted_ft, alpha=0.5, s=30)
    lims = [min(observed_ft.min(), predicted_ft.min()),
            max(observed_ft.max(), predicted_ft.max())]
    ax2.plot(lims, lims, 'k--', linewidth=2)
    ax2.set_xlabel('Observed (feet)')
    ax2.set_ylabel('Predicted (feet)')
    ax2.set_title(f'Scatter Plot (R² = {metrics["R2"]:.3f})')
    ax2.grid(True, alpha=0.3)

    # --- residual series -----------------------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(dates, residuals, 'g-', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=2)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Residual (feet)')
    ax3.set_title(f'Residuals (RMSE = {metrics["RMSE"]:.4f} ft)')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # --- residual histogram --------------------------------------------------
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residual (feet)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residual Distribution')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved figure: {save_path}")
    return fig


# =============================================================================
# 3. MAIN
# =============================================================================
def replot_run(run_dir: str, full_df: pd.DataFrame, target_scaler, title: str = None) -> Dict[str, float]:
    pred_path = os.path.join(run_dir, "hybrid_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"  ⚠ Skipped (no predictions): {pred_path}")
        return {}

    print("=" * 70)
    print(f"     RE-PLOTTING IN FEET: {run_dir}")
    print("=" * 70)

    pdf = pd.read_csv(pred_path).drop_duplicates("time_idx").set_index("time_idx").sort_index()

    # Verify the time_idx alignment before trusting the dates, the same check
    # the comparison notebooks make.
    ref_scaled = full_df.set_index("time_idx").loc[pdf.index, TARGET_COL]
    max_diff = float(np.nanmax(np.abs(pdf["observed"].to_numpy() - ref_scaled.to_numpy())))
    if max_diff > 1e-6:
        raise SystemExit(f"  Observed values do not match time_idx alignment ({max_diff:.2e}).")
    print(f"  Index check: saved observations match the daily series (max diff {max_diff:.1e})")

    dates = full_df.set_index("time_idx").loc[pdf.index, DATE_COL].to_numpy()
    observed_ft = to_feet(pdf["observed"].to_numpy(), target_scaler)
    predicted_ft = to_feet(pdf["predicted"].to_numpy(), target_scaler)

    metrics_ft = calculate_metrics(observed_ft, predicted_ft)
    metrics_scaled = calculate_metrics(pdf["observed"].to_numpy(), pdf["predicted"].to_numpy())

    print(f"\n  Period: {pd.Timestamp(dates.min()).date()} to {pd.Timestamp(dates.max()).date()} "
          f"({metrics_ft['n']} days)")
    print(f"  Observed depth to water: {observed_ft.min():.3f} to {observed_ft.max():.3f} ft")
    print("\n  Metrics                 scaled        feet")
    print("  " + "-" * 42)
    for key, unit in [("RMSE", "ft"), ("MAE", "ft"), ("Bias", "ft")]:
        print(f"  {key:<20} {metrics_scaled[key]:>10.4f}  {metrics_ft[key]:>10.4f} {unit}")
    for key in ["R2", "NSE"]:
        print(f"  {key:<20} {metrics_scaled[key]:>10.4f}  {metrics_ft[key]:>10.4f} (dimensionless)")

    save_path = os.path.join(run_dir, "hybrid_results_feet.png")
    plot_in_feet(dates, observed_ft, predicted_ft, metrics_ft, title or DEFAULT_TITLE, save_path)

    out = {f"{k}_ft": v for k, v in metrics_ft.items()}
    pd.DataFrame([{
        "run_dir": run_dir, "n": metrics_ft["n"],
        "date_start": pd.Timestamp(dates.min()).date(), "date_end": pd.Timestamp(dates.max()).date(),
        "RMSE_ft": metrics_ft["RMSE"], "MAE_ft": metrics_ft["MAE"], "Bias_ft": metrics_ft["Bias"],
        "R2": metrics_ft["R2"], "NSE": metrics_ft["NSE"],
        "RMSE_scaled": metrics_scaled["RMSE"], "MAE_scaled": metrics_scaled["MAE"],
    }]).to_csv(os.path.join(run_dir, "metrics_feet.csv"), index=False)
    print(f"  ✓ Saved metrics: {os.path.join(run_dir, 'metrics_feet.csv')}")

    # Predictions in feet, for anyone plotting these elsewhere
    pdf_ft = pd.DataFrame({
        "time_idx": pdf.index, "date": dates,
        "observed_ft": observed_ft, "predicted_ft": predicted_ft,
        "residual_ft": predicted_ft - observed_ft,
    })
    pdf_ft.to_csv(os.path.join(run_dir, "hybrid_predictions_feet.csv"), index=False)
    print(f"  ✓ Saved predictions: {os.path.join(run_dir, 'hybrid_predictions_feet.csv')}")

    return out


if __name__ == "__main__":
    run_dirs = sys.argv[1:] or ["models/hybrid_tft"]

    full_df, target_scaler = load_reference_series()
    print(f"Target scaler (fitted on train only): mean = {target_scaler.mean_[0]:.4f} ft, "
          f"scale = {target_scaler.scale_[0]:.4f} ft")
    print(f"=> 1 scaled unit = {target_scaler.scale_[0]:.4f} ft\n")

    for run_dir in run_dirs:
        replot_run(run_dir, full_df, target_scaler)

    plt.show()
