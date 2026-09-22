# Hybrid TFT–MODFLOW Groundwater-Level Forecasting

One-day-ahead forecasting of depth to groundwater at a single USGS observation
well, combining a calibrated **MODFLOW-2005** physical groundwater-flow model
with a **Temporal Fusion Transformer (TFT)** trained on ERA5 meteorological
forcing and the observed water-level history.

This repository contains the full pipeline — preprocessing, MODFLOW
calibration, hybrid training, benchmarks, ablations and the statistical tests —
together with every prediction file, metrics table and figure reported in the
manuscript.

---

## 1. The model in brief

The hybrid runs in two stages.

**Stage 1 — MODFLOW-2005 physics prior.** A 2-layer, 15 × 15 finite-difference
groundwater-flow model (500 m cells, unconfined over confined) is built and run
through [FloPy](https://github.com/modflowpy/flopy). Monthly stress periods are
forced with ERA5-derived recharge (a fixed fraction of precipitation, default
0.15) and pumping. Aquifer parameters — hydraulic conductivity, vertical
anisotropy, specific yield, specific storage, boundary heads — are calibrated by
`scipy.optimize.differential_evolution` (Latin-hypercube seeded) against
**training-period observations only**. Simulated heads at the observation cell
are converted to depth below land surface and interpolated to daily resolution,
producing the *physics prior* time series.

**Stage 2 — Hybrid TFT.** A Temporal Fusion Transformer consumes a 90-day
encoder window and predicts one day ahead. Its inputs are:

| Group | Examples |
|---|---|
| Observed water-level history | lags, rolling means, differences |
| Climate forcing (ERA5) | temperature, precipitation, solar radiation, evaporation, wind, soil temperature, snowfall, and their lags / rolling aggregates |
| Calendar | year, month, day-of-year, cyclical sin/cos encodings |
| **MODFLOW physics prior** | `modflow_scaled`, `modflow_lag_1`, `modflow_lag_7`, `modflow_rmean_7`, `modflow_residual_lag_1` |

The TFT's Variable Selection Networks (VSN) then report how much weight the
model actually assigns to each input group — including the physics prior — and
the attention weights show which positions in the 90-day window drive each
forecast. This interpretability is a core contribution, not a by-product.

**Why the prior is only a prior.** The MODFLOW residual is deliberately exposed
only at lag 1 (`modflow_residual_lag_1`). The contemporaneous residual
(`target − modflow_scaled`) and the unscaled `modflow_prediction` column are
explicitly excluded from the feature set, because the former contains the
forecast target. See the leakage notes in §8.

### Data and splits

| | |
|---|---|
| Target | `water_level` — depth to water below land surface, **feet** (larger value = deeper water table) |
| Predictors | ERA5 reanalysis (6-hourly → daily aggregates) plus engineered lag / rolling / calendar features (~207 columns) |
| Record | 2015-04-04 → 2025-12-12, daily |
| Train | 2015-04-04 → 2021-12-31 (2 464 days) |
| Validation | 2022-01-01 → 2023-12-31 (730 days) |
| Test | 2024-01-01 → 2025-12-12 (712 days) |

The split is strictly chronological. All scalers (`StandardScaler`) are fit on
the **train split only** and stored in `data/processed/scalers.joblib`; the same
scaler object is reused everywhere predictions are inverse-transformed back to
feet.

---

## 2. Repository layout

```
├── README.md                          ← this file
├── requirements.txt
├── run_all.ps1                        ← runs the three main stages end to end
│
├── era5_preprocessing.py              ← ERA5 6-hourly → daily aggregates
├── usgs_preprocessing.py              ← USGS water levels → daily, QC'd
├── data_merging.py                    ← join ERA5 + USGS on date
├── feature_engineering.py             ← lags, rolling stats, calendar features
├── data_spliting_and_scalling.py      ← chronological split + train-only scaling
│
├── MODFLOW.py                         ← MODFLOW model construction helpers
├── _run_modflow_vf.py                 ← STAGE 1: MODFLOW calibration → physics prior
├── _run_hybrid_tft.py                 ← STAGE 2: hybrid TFT training + evaluation
│
├── baseline_forecasts.py              ← persistence, seasonal persistence, AR(p)
├── run_ablation_no_modflow.py         ← same TFT, MODFLOW features switched off
├── compare_equivalent_conditions.py   ← scores every model on the same period / resolution
├── plot_results_in_feet.py            ← re-plots any saved run in feet (original units)
│
├── ablation_tests/                    ← MODFLOW ablation ladder, VSN + attention
│   ├── 01_tft_no_modflow.ipynb
│   ├── 02_tft_with_modflow_input.ipynb
│   ├── 03_hybrid_full.ipynb
│   ├── 04_ablation_comparison_and_vsn.ipynb
│   ├── 05_multiseed_runs.ipynb
│   ├── README.md                      ← detailed notes on the ablation design
│   └── results/
│
├── data/
│   ├── raw/                           ← era5_raw.csv, usgs_gwl_raw.csv
│   ├── processed/                     ← era5_daily, usgs_gwl_clean, *_scaled.csv, scalers.joblib
│   ├── merged/merged_dataset.csv
│   └── engineered/engineered_dataset.csv
│
├── models/                            ← one folder per run (predictions, metrics, figures)
│   ├── modflow_enhanced/              ← the reported physics prior
│   ├── modflow_train_only/            ← stricter, train-only-calibrated prior
│   ├── hybrid_tft/                    ← the reported hybrid run
│   └── ablation_no_modflow/           ← TFT with no physics prior
│
└── results/
    ├── baseline_metrics.csv
    ├── equivalent_conditions/         ← like-for-like comparison + bootstrap + DM tests
    └── paper_tables/                  ← paper-ready CSV and LaTeX tables
```

### Notebook variants

The `.py` scripts are authoritative; the notebooks are cell-by-cell copies used
during the manuscript revision, kept so every reported figure stays traceable.

| Notebook | What it is |
|---|---|
| `MODFLOW_VF.ipynb` | Stage 1, notebook form of `_run_modflow_vf.py` |
| `MODFLOW_single_seed_train_only.ipynb` | Stage 1 with the stricter train-only calibration |
| `hybrid_tft_modflow_corrected_VF.ipynb` | **The reported hybrid run** (leakage-corrected evaluation) |
| `hybrid_tft_corrected_modflow_train_only_VF.ipynb` | Same, but using the train-only prior |
| `ftf-modflow_VF.ipynb`, `ftf-modflow_VF_train_only_prior.ipynb`, `hybrid_tft_modflow_corrected.ipynb`, `hybrid_tft_modflow_train_only.ipynb` | Earlier revisions, retained for provenance |
| `paper_comparison_dm_tests.ipynb` | Final benchmark table + Diebold–Mariano tests |

---

## 3. Setup

### Python environment

Developed on **Python 3.11.9** (Windows 10). From a fresh virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

`pytorch_forecasting` pulls in `torch` and `lightning`. Training was done on
**CPU only**; a CUDA build of torch works but is not required.

### MODFLOW-2005 executable

The USGS MODFLOW-2005 distribution is **not** included in this repository (it is
third-party software, ~78 MB). Download release **1.12** from the USGS:

<https://water.usgs.gov/ogw/modflow/MODFLOW.html>

Unzip it so the executable sits at the path the scripts expect:

```
MF2005.1_12/MF2005.1_12/bin/mf2005.exe
```

Or pass your own path via the `exe_name` argument at the bottom of
`_run_modflow_vf.py`. On macOS/Linux, build from the bundled source or install
via `conda install -c conda-forge modflow` and set `exe_name='mf2005'`.

Only **Stage 1** needs MODFLOW. The physics prior it produces
(`models/modflow_enhanced/enhanced_predictions.csv`) is already committed here,
so Stage 2 and everything downstream run without MODFLOW installed.

---

## 4. Running the pipeline

### Quickest path (everything already preprocessed)

`data/processed/` and the MODFLOW prior are committed, so you can go straight to
training the hybrid:

```bash
python _run_hybrid_tft.py
```

### Full pipeline, stage by stage

```bash
# --- Preprocessing (only needed to rebuild data/processed from data/raw) ---
python era5_preprocessing.py            # → data/processed/era5_daily.csv
python usgs_preprocessing.py            # → data/processed/usgs_gwl_clean.csv
python data_merging.py                  # → data/merged/merged_dataset.csv
python feature_engineering.py           # → data/engineered/engineered_dataset.csv
python data_spliting_and_scalling.py    # → data/processed/{train,val,test}_scaled.csv
                                        #   + scalers.joblib, feature_groups.joblib

# --- Stage 1: MODFLOW calibration (needs mf2005.exe) ---
python _run_modflow_vf.py               # → models/modflow_enhanced/enhanced_predictions.csv

# --- Stage 2: hybrid TFT ---
python _run_hybrid_tft.py               # → models/hybrid_tft/

# --- Benchmarks, ablation and statistics ---
python baseline_forecasts.py            # → results/baseline_metrics.csv
python run_ablation_no_modflow.py       # → models/ablation_no_modflow/
python compare_equivalent_conditions.py # → results/equivalent_conditions/

# --- Optional: re-plot any run in feet instead of scaled units ---
python plot_results_in_feet.py models/hybrid_tft
```

**Order matters.** `_run_hybrid_tft.py` reads the MODFLOW prior;
`compare_equivalent_conditions.py` reads both the hybrid and the no-MODFLOW
ablation predictions. Run preprocessing → Stage 1 → Stage 2 → comparisons.

### Unattended run (Windows)

`run_all.ps1` runs the three main stages in order and tees each stage's console
output to `logs/<stage>_<timestamp>.log`, so you can walk away:

```powershell
powershell -ExecutionPolicy Bypass -File run_all.ps1
powershell -ExecutionPolicy Bypass -File run_all.ps1 -Steps baseline   # one stage only
```

Valid `-Steps` values: `baseline`, `modflow`, `hybrid`, `all` (default).

### Runtime expectations

| Stage | Cost |
|---|---|
| Preprocessing | seconds to a few minutes |
| MODFLOW calibration | hours — `differential_evolution` with `maxiter=80`, `popsize=15`, one full MODFLOW run per candidate |
| Hybrid TFT training | hours on CPU — 300 max epochs, early-stopping patience 60, 90-day encoder |
| Benchmarks / comparisons | seconds |

### Reproducibility

Seed 42 is set for `numpy` and `torch` at the top of `_run_hybrid_tft.py`.
Bit-for-bit reproduction across different hardware or library versions is not
guaranteed; `ablation_tests/05_multiseed_runs.ipynb` quantifies the seed-to-seed
spread (see §7) so the results can be read against it.

---

## 5. Key hyperparameters

Set at the bottom of `_run_hybrid_tft.py`:

| Parameter | Value |
|---|---|
| `max_encoder_length` | 90 days |
| `max_prediction_length` | 1 day |
| `hidden_size` | 128 |
| `attention_head_size` | 24 |
| `lstm_layers` | 3 |
| `dropout` | 0.4 |
| `learning_rate` | 1e-4 |
| `batch_size` | 616 |
| `max_epochs` / `patience` | 300 / 60 |
| Loss | `QuantileLoss` |

MODFLOW (in `_run_modflow_vf.py`): 2 layers, 15 × 15 grid, 500 m cells,
`recharge_fraction=0.15`, `reference_elevation=100.0` ft, observation cell at
row 7 / column 7, calibration `maxiter=80`, `popsize=15`.

---

## 6. Outputs

Each training run writes a self-contained folder:

| File | Contents |
|---|---|
| `hybrid_predictions.csv` | `time_idx, observed, predicted` for the test days |
| `hybrid_predictions_feet.csv`, `metrics_feet.csv` | the same, inverse-transformed to feet |
| `hybrid_results.png`, `hybrid_training_history.png` | diagnostics |
| `interpretation/variable_importance.csv` | VSN weights per input (static / encoder / decoder) |
| `interpretation/attention_weights.csv` | attention by position in the encoder window |
| `interpretation/interpretation_*.png` | `pytorch_forecasting` interpretation plots |
| `run_config.json` | configuration, feature list, epochs, metrics (ablation runs) |

Model checkpoints (`hybrid_tft_modflow.pt`) are **not** committed — see §9 — but
every prediction file, metrics table and figure behind the reported numbers is.

`results/` holds the cross-model outputs: `baseline_metrics.csv`,
`equivalent_conditions/` (like-for-like metrics, bootstrap CIs, DM tests) and
`paper_tables/` (paper-ready CSV + LaTeX).

---

## 7. Results summary

**Test period (2024-01-01 → 2025-12-12), daily, in feet.** Metrics are reported
in original units with 95 % moving-block-bootstrap confidence intervals, from
`compare_equivalent_conditions.py` and `baseline_forecasts.py`.

| Model | RMSE (ft) | MAE (ft) | NSE |
|---|---|---|---|
| Hybrid TFT–MODFLOW | **0.01215** | **0.00920** | 0.9935 |
| TFT (no MODFLOW) | 0.01247 | 0.00939 | 0.9931 |
| AR(7) | 0.01266 | 0.00939 | 0.9929 |
| AR(1) | 0.01350 | 0.00978 | 0.9919 |
| Persistence (lag-1) | 0.01352 | 0.00956 | 0.9919 |
| Seasonal persistence (lag-365) | 0.20561 | 0.16757 | −0.8724 |
| MODFLOW alone (interpolated to daily) | 0.60478 | 0.54108 | −15.20 |

**Read these numbers carefully.** At a one-day horizon, persistence already
reaches NSE ≈ 0.99, so the headroom for any model is small. The hybrid's
improvement over the standalone TFT is real but modest: a Diebold–Mariano test
on squared errors gives p = 0.049 (two-sided) over all test days and p = 0.068
on measured days only — borderline, and not significant under absolute-error
loss (p = 0.152). Monthly-resolution differences are not significant at all. The
MODFLOW-alone row is interpolation skill between 24 monthly points, not a
genuine daily simulation, and is labelled as such.

**Ablation ladder** (`ablation_tests/results/tables/`, MSE skill relative to
persistence, mean ± sd over seeds):

| Configuration | Seeds | RMSE (ft) | MSE skill vs persistence |
|---|---|---|---|
| Persistence (lag-1) | — | 0.01352 | 0 % |
| TFT (no MODFLOW) | 1, 2, 3, 42 | 0.01230 ± 0.00009 | 17.3 % ± 1.2 |
| TFT + MODFLOW input only | 42 | 0.01219 | 18.8 % |
| Complete hybrid | 1, 2, 42 | 0.01249 ± 0.00026 | 14.7 % ± 3.6 |

Seed-to-seed spread is comparable to the differences between configurations, so
the ladder should not be over-read.

**VSN importance by variable family** (% of total selection weight):

| Family | TFT (no MODFLOW) | TFT + MODFLOW input | Complete hybrid |
|---|---|---|---|
| Observed water-level history | 47.0 | 70.5 | 41.7 |
| Climate forcing | 17.4 | 9.0 | 27.0 |
| Calendar | 28.0 | 17.2 | 13.8 |
| **MODFLOW (physics prior)** | 0.0 | 0.8 | **14.9** |
| Dataset bookkeeping | 7.5 | 2.5 | 2.7 |

The hybrid does assign the prior meaningful weight (~15 %) rather than ignoring
it. Note that a VSN weight is a variable-*selection* weight: it shows the model
attends to the prior, but does not by itself prove the output obeys the
groundwater-flow equation.

---

## 8. Known caveats and leakage notes

Stated openly because they affect how the results should be read.

1. **TFT evaluation window (fixed).** An earlier version built the test
   `TimeSeriesDataSet` spanning nearly the whole record. The corrected pipeline
   in `_run_hybrid_tft.py` and `hybrid_tft_modflow_corrected_VF.ipynb` scores
   test days only. Older notebook variants are kept for provenance and their
   numbers are superseded.
2. **MODFLOW residual.** Only `modflow_residual_lag_1` is an input; the
   contemporaneous residual and the unscaled `modflow_prediction` column are
   explicitly excluded (see `_identify_features` in `_run_hybrid_tft.py`).
3. **Residual trend adjustment in the prior.** In `MODFLOW_VF.ipynb` the
   calibration objective is restricted to the training period, but the
   `trend_adjustment` slope is still fitted on the full observed record — a
   small remaining leak, flagged in that notebook's own `LEAKAGE NOTE`.
   `MODFLOW_single_seed_train_only.ipynb` and
   `models/modflow_train_only/train_only_predictions.csv` provide the stricter
   alternative; the hybrid can be re-run against it by swapping `modflow_path`.
4. **Monthly-to-daily interpolation.** The prior is interpolated from monthly
   stress periods, so a given day's value reflects simulated recharge up to
   roughly a month later. That is weather forcing rather than observed
   groundwater levels — not target leakage, but worth one sentence in Methods.
5. **Single well, single site.** All results are for one observation well. No
   claim of spatial generalisation is made.
6. **Single seed for the headline run.** The reported hybrid uses seed 42;
   `ablation_tests/05_multiseed_runs.ipynb` quantifies the spread.

---

## 9. What is and is not in this repository

**Included:** all code and notebooks; raw, merged, engineered and scaled data;
fitted scalers; calibrated MODFLOW parameters and the resulting physics prior;
every prediction CSV, metrics table, VSN / attention table and figure behind the
reported results.

**Excluded** (see `.gitignore`):

| Excluded | Why |
|---|---|
| `MF2005.1_12/` | Third-party USGS distribution — download link in §3 |
| `*.pt`, `*.zip` | Trained checkpoints, regenerated by the training scripts |
| MODFLOW workspace scratch (`*.hds`, `*.cbc`, `*.nam`, …) | Regenerated on every calibration run |
| `logs/`, `lightning_logs/`, `__pycache__/` | Run telemetry and caches |
| `data/_pre_leakage_fix_backup/` | Superseded working copies |

---

## 10. Citation and contact

If you use this code, please cite the accompanying manuscript. Questions about
reproduction are welcome via the repository's issue tracker.
