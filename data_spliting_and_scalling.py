import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import joblib
import os

# Windows consoles default to the cp1252 codepage, which can't encode the
# "✓"/"█" characters used in this file's progress prints. Force UTF-8 stdout
# so the script runs the same everywhere, regardless of console codepage.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# =============================================================================
# 1. LOAD ENGINEERED DATA
# =============================================================================
def load_engineered_data(filepath: str) -> pd.DataFrame:
    """
    Load the engineered dataset and prepare for splitting. 
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print("="*70)
    print("              LOADING ENGINEERED DATASET")
    print("="*70)
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Missing values: {df. isnull().sum().sum()}")
    
    return df


# =============================================================================
# 2. HANDLE MISSING VALUES FROM LAG OPERATIONS
# =============================================================================
def handle_missing_values(df: pd. DataFrame, 
                          max_lag: int = 90,
                          strategy: str = 'drop_initial') -> pd.DataFrame:
    """
    Handle missing values created by lag and diff operations.
    
    Strategies:
    - 'drop_initial':  Drop first max_lag rows (recommended)
    - 'forward_fill': Forward fill then backward fill
    - 'interpolate': Linear interpolation
    """
    print("\n" + "="*70)
    print("              HANDLING MISSING VALUES")
    print("="*70)
    
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before:  {missing_before}")
    
    if strategy == 'drop_initial':
        # Drop first rows where lag features are NaN
        df_clean = df.iloc[max_lag: ].reset_index(drop=True)
        print(f"Strategy:  Dropping first {max_lag} rows (lag initialization period)")
        
    elif strategy == 'forward_fill': 
        df_clean = df. fillna(method='ffill').fillna(method='bfill')
        print("Strategy: Forward fill then backward fill")
        
    elif strategy == 'interpolate': 
        df_clean = df.interpolate(method='linear', limit_direction='both')
        print("Strategy: Linear interpolation")
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values after:  {missing_after}")
    print(f"Records remaining: {len(df_clean)}")
    print(f"New date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    
    return df_clean


# =============================================================================
# 3. TEMPORAL TRAIN/VALIDATION/TEST SPLIT
# =============================================================================
def temporal_train_val_test_split(
    df: pd. DataFrame,
    train_end: str = '2021-12-31',
    val_end: str = '2023-12-31',
    date_col: str = 'date'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd. DataFrame]: 
    """
    Split data temporally into train, validation, and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The engineered dataset
    train_end :  str
        Last date for training set (inclusive)
    val_end : str
        Last date for validation set (inclusive)
    date_col :  str
        Name of the date column
    
    Returns:
    --------
    Tuple of (train_df, val_df, test_df)
    """
    print("\n" + "="*70)
    print("              TEMPORAL DATA SPLITTING")
    print("="*70)
    
    # Convert to datetime if needed
    df[date_col] = pd.to_datetime(df[date_col])
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    
    # Split the data
    train_df = df[df[date_col] <= train_end]. copy()
    val_df = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test_df = df[df[date_col] > val_end].copy()
    
    # Calculate statistics
    total_days = len(df)
    train_pct = len(train_df) / total_days * 100
    val_pct = len(val_df) / total_days * 100
    test_pct = len(test_df) / total_days * 100
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    SPLIT SUMMARY                                │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Dataset     │  Records  │  Percentage  │  Date Range          │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  TRAIN       │  {len(train_df):>6}   │   {train_pct:>5.1f}%     │  {train_df[date_col]. min().date()} to {train_df[date_col].max().date()}  │")
    print(f"│  VALIDATION  │  {len(val_df):>6}   │   {val_pct:>5.1f}%     │  {val_df[date_col]. min().date()} to {val_df[date_col]. max().date()}  │")
    print(f"│  TEST        │  {len(test_df):>6}   │   {test_pct:>5.1f}%     │  {test_df[date_col].min().date()} to {test_df[date_col].max().date()}  │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  TOTAL       │  {total_days:>6}   │   100.0%     │                          │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    return train_df, val_df, test_df


# =============================================================================
# 4. DEFINE FEATURE GROUPS
# =============================================================================
def define_feature_groups(df:  pd.DataFrame) -> Dict[str, List[str]]: 
    """
    Define different feature groups for scaling and modeling.
    """
    all_columns = df.columns. tolist()
    
    # Columns to exclude from scaling
    exclude_cols = [
        'date',              # Date column (not a feature)
        'is_interpolated',   # Flag column
        'year'               # Keep year as is for potential grouping
    ]
    
    # Target variable
    target_col = 'water_level'
    
    # Categorical columns (don't scale these, or use different encoding)
    categorical_cols = [
        'month', 'quarter', 'season', 'day_of_week', 'is_weekend',
        'is_hot_day', 'is_cold_day', 'is_wet_day', 'is_dry_day',
        'significant_precip'
    ]
    
    # Numerical columns to scale
    numerical_cols = [
        col for col in all_columns 
        if col not in exclude_cols + categorical_cols + [target_col]
        and col in df.select_dtypes(include=[np.number]).columns
    ]
    
    feature_groups = {
        'target': target_col,
        'exclude': exclude_cols,
        'categorical':  [c for c in categorical_cols if c in all_columns],
        'numerical': numerical_cols,
        'all_features': [c for c in numerical_cols + categorical_cols if c in all_columns]
    }
    
    print("\n" + "="*70)
    print("              FEATURE GROUPS")
    print("="*70)
    print(f"Target variable: {target_col}")
    print(f"Excluded columns: {len(exclude_cols)}")
    print(f"Categorical features: {len(feature_groups['categorical'])}")
    print(f"Numerical features: {len(feature_groups['numerical'])}")
    print(f"Total features for modeling: {len(feature_groups['all_features'])}")
    
    return feature_groups


# =============================================================================
# 5. SCALING FUNCTIONS
# =============================================================================
def scale_features(
    train_df: pd. DataFrame,
    val_df: pd. DataFrame,
    test_df: pd. DataFrame,
    feature_groups: Dict[str, List[str]],
    scaler_type: str = 'standard',
    scale_target: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd. DataFrame, Dict]: 
    """
    Scale features using fit on training data only.
    
    Parameters:
    -----------
    train_df, val_df, test_df : pd. DataFrame
        Split datasets
    feature_groups : dict
        Dictionary defining feature groups
    scaler_type : str
        Type of scaler:  'standard', 'minmax', or 'robust'
    scale_target :  bool
        Whether to scale the target variable
    
    Returns:
    --------
    Tuple of (train_scaled, val_scaled, test_scaled, scalers_dict)
    """
    print("\n" + "="*70)
    print("              FEATURE SCALING")
    print("="*70)
    
    # Choose scaler
    scalers = {
        'standard':  StandardScaler(),    # Z-score:  (x - mean) / std
        'minmax': MinMaxScaler(),         # Scale to [0, 1]
        'robust':  RobustScaler()          # Uses median and IQR (robust to outliers)
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"scaler_type must be one of {list(scalers. keys())}")
    
    print(f"Scaler type: {scaler_type. upper()}")
    
    # Initialize scalers
    feature_scaler = scalers[scaler_type]
    target_scaler = StandardScaler() if scale_target else None
    
    # Get column lists
    numerical_cols = feature_groups['numerical']
    target_col = feature_groups['target']
    
    # Copy dataframes
    train_scaled = train_df. copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    
    # Scale numerical features
    print(f"\nScaling {len(numerical_cols)} numerical features...")
    
    # Fit on training data only
    feature_scaler.fit(train_df[numerical_cols])
    
    # Transform all sets
    train_scaled[numerical_cols] = feature_scaler.transform(train_df[numerical_cols])
    val_scaled[numerical_cols] = feature_scaler.transform(val_df[numerical_cols])
    test_scaled[numerical_cols] = feature_scaler.transform(test_df[numerical_cols])
    
    # Scale target variable (optional but recommended)
    if scale_target: 
        print(f"Scaling target variable:  {target_col}")
        
        # Fit on training data only
        target_scaler. fit(train_df[[target_col]])
        
        # Transform all sets
        train_scaled[target_col] = target_scaler.transform(train_df[[target_col]])
        val_scaled[target_col] = target_scaler.transform(val_df[[target_col]])
        test_scaled[target_col] = target_scaler.transform(test_df[[target_col]])
    
    # Store scalers for later inverse transformation
    scalers_dict = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'numerical_cols': numerical_cols,
        'target_col': target_col,
        'scaler_type': scaler_type
    }
    
    print("\n✓ Scaling complete!")
    print(f"  - Feature scaler fitted on {len(train_df)} training samples")
    print(f"  - Validation set transformed: {len(val_df)} samples")
    print(f"  - Test set transformed: {len(test_df)} samples")
    
    return train_scaled, val_scaled, test_scaled, scalers_dict


# =============================================================================
# 6. INVERSE SCALING (for predictions)
# =============================================================================
def inverse_scale_predictions(
    predictions: np.ndarray,
    scalers_dict: Dict
) -> np.ndarray:
    """
    Inverse transform predictions back to original scale. 
    """
    target_scaler = scalers_dict['target_scaler']
    
    if target_scaler is None:
        return predictions
    
    # Reshape if needed
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    return target_scaler. inverse_transform(predictions).flatten()


def inverse_scale_target(
    df: pd.DataFrame,
    scalers_dict: Dict
) -> pd.DataFrame:
    """
    Inverse transform the target column in a dataframe.
    """
    df = df.copy()
    target_col = scalers_dict['target_col']
    target_scaler = scalers_dict['target_scaler']
    
    if target_scaler is not None:
        df[target_col] = target_scaler.inverse_transform(df[[target_col]])
    
    return df


# =============================================================================
# 7. VERIFICATION AND VISUALIZATION
# =============================================================================
def verify_scaling(
    train_original: pd.DataFrame,
    train_scaled: pd. DataFrame,
    feature_groups: Dict[str, List[str]],
    n_features: int = 5
) -> None:
    """
    Verify scaling by comparing statistics before and after. 
    """
    print("\n" + "="*70)
    print("              SCALING VERIFICATION")
    print("="*70)
    
    numerical_cols = feature_groups['numerical'][: n_features]
    target_col = feature_groups['target']
    
    print("\nNumerical Features (before vs after scaling):")
    print("-" * 70)
    print(f"{'Feature':<35} {'Original Mean': >12} {'Scaled Mean': >12}")
    print("-" * 70)
    
    for col in numerical_cols: 
        orig_mean = train_original[col].mean()
        scaled_mean = train_scaled[col].mean()
        print(f"{col:<35} {orig_mean: >12.4f} {scaled_mean:>12.4f}")
    
    print("-" * 70)
    print(f"\nTarget Variable:  {target_col}")
    print(f"  Original - Mean: {train_original[target_col].mean():.4f}, Std: {train_original[target_col]. std():.4f}")
    print(f"  Scaled   - Mean: {train_scaled[target_col].mean():.4f}, Std: {train_scaled[target_col]. std():.4f}")


def visualize_split(
    train_df: pd. DataFrame,
    val_df: pd. DataFrame,
    test_df: pd. DataFrame,
    target_col: str = 'water_level',
    date_col: str = 'date',
    save_path: str = None
) -> None:
    """
    Visualize the temporal split of the data.
    """
    fig, axes = plt. subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Full timeline with split regions
    ax1 = axes[0]
    ax1.plot(train_df[date_col], train_df[target_col], 
             color='blue', label='Train', alpha=0.7)
    ax1.plot(val_df[date_col], val_df[target_col], 
             color='orange', label='Validation', alpha=0.7)
    ax1.plot(test_df[date_col], test_df[target_col], 
             color='green', label='Test', alpha=0.7)
    
    # Add vertical lines at split points
    ax1.axvline(x=train_df[date_col].max(), color='red', linestyle='--', 
                label='Train/Val Split', linewidth=2)
    ax1.axvline(x=val_df[date_col].max(), color='purple', linestyle='--', 
                label='Val/Test Split', linewidth=2)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{target_col} (feet below surface)')
    ax1.set_title('Temporal Train/Validation/Test Split')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution comparison
    ax2 = axes[1]
    ax2.hist(train_df[target_col], bins=50, alpha=0.5, label='Train', color='blue', density=True)
    ax2.hist(val_df[target_col], bins=50, alpha=0.5, label='Validation', color='orange', density=True)
    ax2.hist(test_df[target_col], bins=50, alpha=0.5, label='Test', color='green', density=True)
    ax2.set_xlabel(f'{target_col} (feet below surface)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution Comparison Across Splits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
    
    plt.show()


# =============================================================================
# 8. SAVE PROCESSED DATA
# =============================================================================
def save_processed_data(
    train_scaled: pd.DataFrame,
    val_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    scalers_dict: Dict,
    output_dir: str = 'data/processed'
) -> None:
    """
    Save processed datasets and scalers.
    """
    print("\n" + "="*70)
    print("              SAVING PROCESSED DATA")
    print("="*70)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_scaled.to_csv(f'{output_dir}/train_scaled.csv', index=False)
    val_scaled.to_csv(f'{output_dir}/val_scaled.csv', index=False)
    test_scaled.to_csv(f'{output_dir}/test_scaled.csv', index=False)

    print(f"✓ Saved train_scaled.csv ({len(train_scaled)} records)")
    print(f"✓ Saved val_scaled.csv ({len(val_scaled)} records)")
    print(f"✓ Saved test_scaled.csv ({len(test_scaled)} records)")

    # Save scalers
    joblib.dump(scalers_dict, f'{output_dir}/scalers.joblib')
    print(f"✓ Saved scalers.joblib")

    # Save feature groups
    joblib.dump(scalers_dict['feature_groups'], f'{output_dir}/feature_groups.joblib')
    print(f"✓ Saved feature_groups.joblib")


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================
def run_splitting_and_scaling_pipeline(
    input_filepath: str = 'data/engineered/engineered_dataset.csv',
    output_dir: str = 'data/processed',
    train_end:  str = '2021-12-31',
    val_end: str = '2023-12-31',
    max_lag: int = 90,
    scaler_type: str = 'standard',
    scale_target: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run the complete splitting and scaling pipeline. 
    """
    print("\n" + "█"*70)
    print("       DATA SPLITTING AND SCALING PIPELINE")
    print("█"*70)
    
    # Step 1: Load data
    df = load_engineered_data(input_filepath)
    
    # Step 2: Handle missing values
    df_clean = handle_missing_values(df, max_lag=max_lag, strategy='drop_initial')
    
    # Step 3: Define feature groups
    feature_groups = define_feature_groups(df_clean)
    
    # Step 4:  Temporal split
    train_df, val_df, test_df = temporal_train_val_test_split(
        df_clean,
        train_end=train_end,
        val_end=val_end
    )
    
    # Step 5: Scale features
    train_scaled, val_scaled, test_scaled, scalers_dict = scale_features(
        train_df, val_df, test_df,
        feature_groups,
        scaler_type=scaler_type,
        scale_target=scale_target
    )
    
    # Add feature groups to scalers_dict for convenience
    scalers_dict['feature_groups'] = feature_groups
    
    # Step 6: Verify scaling
    verify_scaling(train_df, train_scaled, feature_groups)
    
    # Step 7: Visualize split
    visualize_split(train_df, val_df, test_df, save_path=f'{output_dir}/temporal_split.png')
    
    # Step 8: Save processed data
    save_processed_data(train_scaled, val_scaled, test_scaled, scalers_dict, output_dir)
    
    print("\n" + "█"*70)
    print("       PIPELINE COMPLETE!")
    print("█"*70)
    
    return train_scaled, val_scaled, test_scaled, scalers_dict


# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__": 
    
    # Run the pipeline
    train_scaled, val_scaled, test_scaled, scalers_dict = run_splitting_and_scaling_pipeline(
        input_filepath='data/engineered/engineered_dataset.csv',
        output_dir='data/processed',
        train_end='2021-12-31',    # ~7 years for training
        val_end='2023-12-31',       # ~2 years for validation
        max_lag=90,                  # Maximum lag used in feature engineering
        scaler_type='standard',     # StandardScaler (z-score normalization)
        scale_target=True           # Scale target variable too
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("              FINAL DATASET SUMMARY")
    print("="*70)
    print(f"\nTrain set shape: {train_scaled.shape}")
    print(f"Validation set shape: {val_scaled.shape}")
    print(f"Test set shape: {test_scaled.shape}")
    print(f"\nScaler type: {scalers_dict['scaler_type']}")
    print(f"Number of numerical features scaled: {len(scalers_dict['numerical_cols'])}")