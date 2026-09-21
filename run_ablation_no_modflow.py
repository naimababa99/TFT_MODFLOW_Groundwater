"""
Ablation run for R3 Major 6 (physical-consistency reviewer comment): trains
a standalone TFT with the exact same architecture/hyperparameters/corrected
evaluation harness as `_run_hybrid_tft.py`'s main run, but with
`include_modflow_features=False` so no MODFLOW-derived column ever enters
the model's inputs. Compared against the Hybrid model's corrected test
metrics, this is the "TFT without MODFLOW" arm of the ablation.

Run with the environment that has pytorch_forecasting installed, e.g.:
    C:\\Users\\pctrema\\anaconda3\\python.exe run_ablation_no_modflow.py
"""

from _run_hybrid_tft import run_hybrid_tft_modflow_pipeline

if __name__ == "__main__":
    predictions, actuals, metrics, model = run_hybrid_tft_modflow_pipeline(
        # Data paths (flat layout: data/ and models/ are siblings of this file)
        train_path='data/processed/train_scaled.csv',
        val_path='data/processed/val_scaled.csv',
        test_path='data/processed/test_scaled.csv',
        modflow_path='models/modflow_enhanced/enhanced_predictions.csv',
        scaler_path='data/processed/scalers.joblib',

        # Output -- separate directory so this never overwrites the hybrid model
        model_dir='models/ablation_no_modflow',

        # Data columns
        target_col='water_level',
        date_col='date',
        modflow_pred_col='simulated_depth_ft',

        # Model parameters -- identical to the main hybrid run for a fair
        # ablation comparison
        max_encoder_length=90,
        max_prediction_length=1,
        hidden_size=128,
        attention_head_size=24,
        lstm_layers=3,
        dropout=0.4,
        learning_rate=0.0001,
        batch_size=616,
        max_epochs=300,
        patience=60,

        # The ablation switch
        include_modflow_features=False,
    )

    if metrics:
        print("\n" + "=" * 50)
        print("TFT WITHOUT MODFLOW -- FINAL RESULTS")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
