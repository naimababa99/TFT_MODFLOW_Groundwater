# `ablation_tests/` — MODFLOW ablation ladder, VSN weights and attention

Written for: whoever runs these notebooks and writes the reviewer response (the authors).

This folder answers one reviewer comment:

> "Using MODFLOW predictions as a TFT input does not automatically ensure that the final
> predictions remain physically consistent. The TFT may assign limited importance to the
> MODFLOW variable or override its influence through correlations with observed
> groundwater-level lags. An ablation experiment should compare: TFT without MODFLOW input;
> TFT with MODFLOW input; the complete hybrid configuration. The manuscript should also
> present the Variable Selection Network weights and attention results. VSN-based
> interpretation is identified as a major contribution, but detailed feature-importance
> results and their hydrogeological interpretation are currently insufficient."

Nothing here has been run yet. The notebooks were generated from
`../hybrid_tft_modflow_corrected_VF.ipynb` and carry all of its leakage fixes unchanged.

## Contents

| Notebook | Configuration | MODFLOW inputs the model can see | Results folder |
|---|---|---|---|
| `01_tft_no_modflow.ipynb` | `modflow_feature_mode='none'` | none | `results/tft_no_modflow/` |
| `02_tft_with_modflow_input.ipynb` | `modflow_feature_mode='prediction_only'` | `modflow_scaled` | `results/tft_modflow_input_only/` |
| `03_hybrid_full.ipynb` | `modflow_feature_mode='full'` | `modflow_scaled`, `modflow_lag_1`, `modflow_lag_7`, `modflow_rmean_7`, `modflow_residual_lag_1` | `results/hybrid_full/` |
| `04_ablation_comparison_and_vsn.ipynb` | — | reads the three runs above | `results/tables/` |

Notebook 3 is the paper's reported configuration. It is re-run here rather than reusing
`models/hybrid_tft/` so that all three rungs come from the same code, the same splits and
the same seed, and land in comparable folders.

## How to run

1. Use an environment with `pytorch_forecasting` installed (notebooks 01–03).
   Notebook 04 needs only pandas, numpy, scipy, joblib and matplotlib.
2. Start the kernel in either `code_version_2/` or `code_version_2/ablation_tests/` — the
   **Path setup** cell steps up one level when needed, so all the `data/…` and `models/…`
   paths work either way.
3. Run `01`, `02`, `03` in any order (they are independent), then `04`.
4. Training is CPU-only on the original machine and slow: 300 max epochs, patience 60,
   encoder length 90. Budget accordingly, and expect each run to take about as long as the
   original hybrid run did.

Each of 01–03 writes into its results folder:

| File | Used for |
|---|---|
| `hybrid_predictions.csv` | test-day predictions — the accuracy ladder and the DM tests |
| `run_config.json` | which configuration produced the folder, its feature list, epochs, metrics |
| `hybrid_results.png`, `hybrid_training_history.png` | per-run diagnostics |
| `hybrid_tft_modflow.pt` | trained weights |
| `interpretation/variable_importance.csv` | VSN weights per input (static/encoder/decoder) |
| `interpretation/attention_weights.csv` | attention by position in the encoder window |
| `interpretation/interpretation_*.png` | pytorch_forecasting's own interpretation plots |

Notebook 04 then writes CSV + LaTeX tables and two figures into `results/tables/`:
the accuracy ladder (all days and measured days only), the Diebold–Mariano tests between
every pair of configurations, VSN importance per variable and per variable family,
`vsn_importance_by_family.png` and `attention_over_lead_time.png`.

## Reusing the existing hybrid run instead of running `03`

`results/hybrid_full/` currently holds a **copy of `models/hybrid_tft/`** — the completed run
of `../hybrid_tft_modflow_corrected_VF.ipynb`, which is the same configuration notebook 03
would produce. Notebook 04 was tested against this copy and works: it loads the run, scores
it, runs the DM tests against persistence, and builds the VSN tables and figure.

Two things to know about the reuse path:

- `run_config.json` in that folder is **hand-written**, not produced by the pipeline. Its
  `source` field says so. The feature list in it was verified against `_identify_features`
  in mode `'full'`; the metric fields are left empty because the pipeline records them in
  scaled units.
- `interpretation/attention_weights.csv` does not exist for that run (it predates the
  addition), so notebook 04 omits it from the attention figure. `interpretation_attention.png`
  from the original run is still there and usable in the paper. To get the CSV without
  retraining, load `hybrid_tft_modflow.pt` into a model rebuilt by `create_datasets` and call
  `extract_and_save_interpretation` on the test loader — that is inference only, no training.

If you would rather have all three rungs produced by one code path in one sitting, delete the
copy and run `03` normally.

## What differs from `hybrid_tft_modflow_corrected_VF.ipynb`

Deliberately minimal — the ablation is only interpretable if the feature set is the single
thing that varies:

1. The boolean `include_modflow_features` switch became a three-level
   `modflow_feature_mode` (`'none'` / `'prediction_only'` / `'full'`), validated in
   `HybridTFTConfig.__init__`, with the feature list for each mode in
   `HybridDataPreparator._identify_features`.
2. A **Path setup** cell, so the notebooks can live in this subfolder.
3. The pipeline writes `run_config.json`.
4. `extract_and_save_interpretation` additionally saves `attention_weights.csv`
   (it already saved `variable_importance.csv` and the PNGs).

Architecture, hyperparameters, seed (42), splits, the MODFLOW prior file, the training loop
and every leakage fix are untouched.

## Reading the results

The reviewer's worry is that the model may ignore the prior, so the two halves must be read
together:

- **Accuracy ladder** (`Skill vs no-MODFLOW (%)`): does adding the prior improve forecasts?
- **VSN share** (`MODFLOW (physics prior)` row): does the model assign the prior any weight,
  or do the observed water-level lags dominate?

If the MODFLOW share is small **and** the skill gain is near zero or negative, the honest
conclusion is that the prior is not contributing at this horizon, and the manuscript should
say so instead of claiming physical consistency. That is a legitimate finding: at a one-day
horizon, persistence alone reaches NSE ≈ 0.99, so there is very little headroom for a
monthly-resolution physics prior to add. Notebook 04's section 10 prints wording for either
outcome.

Also note: a VSN weight is a variable-selection weight, not a causal or physical-consistency
measure. A high MODFLOW share shows the model attends to the prior; it does not by itself
prove the output obeys the groundwater flow equation. If the manuscript wants a physical
consistency claim, that needs a separate check (for example, whether the hybrid's forecasts
respect a water-balance or monotonicity constraint the prior satisfies).

## Caveats worth stating in the response letter

- **Single seed.** All three runs use seed 42, as the paper's run did. Differences between
  rungs that are small relative to seed-to-seed variation should not be over-interpreted.
  Re-running each notebook under a few seeds and reporting mean ± spread would be stronger,
  and needs only the `set_seed(42)` call in the imports cell changed.
- **Which MODFLOW prior.** All three notebooks use
  `models/modflow_enhanced/enhanced_predictions.csv`, matching the paper's reported run.
  A commented alternative line in each "Run the pipeline" cell switches to
  `models/modflow_train_only/train_only_predictions.csv` (calibrated on training-period
  observations only). If you report the stricter protocol, re-run all three with the prior
  swapped and change the results folder names so the two sets never mix.
- **The MODFLOW prior itself has a small remaining leak**: in `MODFLOW_VF.ipynb` the
  calibration objective is restricted to the training period, but the `trend_adjustment`
  slope is still fitted on the full observed record (see that notebook's own
  `LEAKAGE NOTE`).
- **Monthly-to-daily interpolation.** The prior is interpolated from monthly stress periods,
  so a given day's value reflects simulated recharge up to roughly a month later. That is
  weather forcing rather than observed groundwater levels, so it is not target leakage, but
  it is worth one sentence in the Methods section.
