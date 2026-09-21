import sys
import pandas as pd
import numpy as np
from typing import List, Tuple

# Windows consoles default to the cp1252 codepage, which can't encode the
# "✓" characters used in this file's progress prints. Force UTF-8 stdout so
# the script runs the same everywhere, regardless of console codepage.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load merged dataset and prepare for feature engineering. 
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Dataset loaded:  {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    return df


# =============================================================================
# 1. TEMPORAL FEATURES
# =============================================================================
def add_temporal_features(df:  pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the date column.
    These capture seasonal and cyclical patterns in groundwater levels.
    """
    df = df.copy()
    
    # Basic temporal components
    df['year'] = df['date'].dt.year
    df['month'] = df['date']. dt.month
    df['day'] = df['date']. dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df['date']. dt.dayofweek  # 0=Monday, 6=Sunday
    df['quarter'] = df['date'].dt.quarter
    
    # Season (for Northern Hemisphere - Nevada)
    def get_season(month):
        if month in [12, 1, 2]: 
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else: 
            return 3  # Fall
    
    df['season'] = df['month'].apply(get_season)
    
    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Days since start (for trend capture)
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    print(f"✓ Added {10} temporal features")
    
    return df


# =============================================================================
# 2. CYCLICAL ENCODING
# =============================================================================
def add_cyclical_features(df: pd. DataFrame) -> pd.DataFrame:
    """
    Encode cyclical features using sine and cosine transformations.
    This helps models understand that December is close to January.
    """
    df = df.copy()
    
    # Day of year (period = 365)
    df['day_of_year_sin'] = np. sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Month (period = 12)
    df['month_sin'] = np.sin(2 * np. pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Week of year (period = 52)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Day of week (period = 7)
    df['day_of_week_sin'] = np.sin(2 * np. pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    print(f"✓ Added 8 cyclical features")
    
    return df


# =============================================================================
# 3. LAG FEATURES
# =============================================================================
def add_lag_features(df: pd. DataFrame, 
                     target_col: str = 'water_level',
                     climate_cols: List[str] = None,
                     target_lags: List[int] = None,
                     climate_lags: List[int] = None) -> pd.DataFrame:
    """
    Add lagged features for target variable and climate variables.
    Lagged features are crucial for time series prediction.
    """
    df = df. copy()
    
    # Default lag periods
    if target_lags is None: 
        target_lags = [1, 2, 3, 7, 14, 30, 60, 90]  # days
    
    if climate_lags is None:
        climate_lags = [1, 3, 7, 14, 30]  # days
    
    if climate_cols is None:
        climate_cols = [
            'temp_celsius_mean', 
            'total_precipitation_sum',
            'evaporation_sum', 
            'solar_radiation_MJ',
            'wind_speed',
            'soil_temperature_level_1_mean'
        ]
    
    # Target variable lags (groundwater level)
    for lag in target_lags:
        df[f'{target_col}_lag_{lag}d'] = df[target_col].shift(lag)
    
    # Climate variable lags
    for col in climate_cols: 
        for lag in climate_lags: 
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
    
    n_target_lags = len(target_lags)
    n_climate_lags = len(climate_cols) * len(climate_lags)
    
    print(f"✓ Added {n_target_lags} target lag features")
    print(f"✓ Added {n_climate_lags} climate lag features")
    
    return df


# =============================================================================
# 4. ROLLING STATISTICS
# =============================================================================
def add_rolling_features(df: pd. DataFrame,
                         target_col: str = 'water_level',
                         climate_cols: List[str] = None,
                         windows: List[int] = None) -> pd.DataFrame:
    """
    Add rolling window statistics to capture trends and variability.
    """
    df = df.copy()
    
    if windows is None:
        windows = [7, 14, 30, 60, 90]  # days
    
    if climate_cols is None: 
        climate_cols = [
            'temp_celsius_mean',
            'total_precipitation_sum',
            'evaporation_sum',
            'solar_radiation_MJ'
        ]
    
    feature_count = 0

    # Rolling statistics for target variable
    # NOTE: computed on the lag-1 series so that, at row t, every rolling
    # window covers only [t-window, t-1] -- i.e. strictly before the
    # forecast origin. Using df[target_col] directly here would leak the
    # current-day (and, for a 1-day horizon, the target) value into the
    # window since pandas .rolling(min_periods=1) includes the current row.
    target_shifted = df[target_col].shift(1)
    for window in windows:
        # Mean
        df[f'{target_col}_rolling_mean_{window}d'] = (
            target_shifted.rolling(window=window, min_periods=1).mean()
        )
        # Standard deviation
        df[f'{target_col}_rolling_std_{window}d'] = (
            target_shifted.rolling(window=window, min_periods=1).std()
        )
        # Min and Max
        df[f'{target_col}_rolling_min_{window}d'] = (
            target_shifted.rolling(window=window, min_periods=1).min()
        )
        df[f'{target_col}_rolling_max_{window}d'] = (
            target_shifted.rolling(window=window, min_periods=1).max()
        )
        # Range
        df[f'{target_col}_rolling_range_{window}d'] = (
            df[f'{target_col}_rolling_max_{window}d'] -
            df[f'{target_col}_rolling_min_{window}d']
        )
        feature_count += 5
    
    # Rolling statistics for climate variables
    for col in climate_cols: 
        for window in windows:
            # Mean
            df[f'{col}_rolling_mean_{window}d'] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
            # Sum (useful for precipitation)
            df[f'{col}_rolling_sum_{window}d'] = (
                df[col].rolling(window=window, min_periods=1).sum()
            )
            # Standard deviation
            df[f'{col}_rolling_std_{window}d'] = (
                df[col]. rolling(window=window, min_periods=1).std()
            )
            feature_count += 3
    
    print(f"✓ Added {feature_count} rolling statistics features")
    
    return df


# =============================================================================
# 5. CUMULATIVE FEATURES
# =============================================================================
def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative features within year for tracking seasonal accumulation.
    """
    df = df.copy()
    
    # Group by year for within-year cumulative sums
    df['precip_cumsum_yearly'] = df.groupby('year')['total_precipitation_sum'].cumsum()
    df['evap_cumsum_yearly'] = df.groupby('year')['evaporation_sum'].cumsum()
    df['snowfall_cumsum_yearly'] = df. groupby('year')['snowfall_sum'].cumsum()
    df['solar_cumsum_yearly'] = df.groupby('year')['solar_radiation_MJ']. cumsum()
    
    # Cumulative water balance (simplified)
    df['water_balance_cumsum_yearly'] = (
        df['precip_cumsum_yearly'] + 
        df['snowfall_cumsum_yearly'] - 
        df['evap_cumsum_yearly']
    )
    
    # Days since last significant precipitation (> threshold)
    precip_threshold = df['total_precipitation_sum'].quantile(0.75)
    df['significant_precip'] = (df['total_precipitation_sum'] > precip_threshold).astype(int)
    
    # Calculate days since last significant precipitation
    df['days_since_significant_precip'] = 0
    days_counter = 0
    for i in range(len(df)):
        if df.loc[i, 'significant_precip'] == 1:
            days_counter = 0
        else:
            days_counter += 1
        df.loc[i, 'days_since_significant_precip'] = days_counter
    
    print(f"✓ Added 6 cumulative features")
    
    return df


# =============================================================================
# 6. DERIVED HYDROLOGICAL FEATURES
# =============================================================================
def add_hydrological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features relevant to groundwater hydrology.
    """
    df = df.copy()
    
    # Water balance proxy (Precipitation - Evaporation)
    df['water_balance'] = df['total_precipitation_sum'] - np.abs(df['evaporation_sum'])
    
    # Potential recharge indicator (precip + snowmelt - evap)
    # Snowmelt approximation:  when soil temp > 0°C and there was recent snow
    df['soil_temp_celsius'] = df['soil_temperature_level_1_mean'] - 273.15
    df['potential_snowmelt'] = np.where(
        (df['soil_temp_celsius'] > 0) & (df['snowfall_sum']. shift(1) > 0),
        df['snowfall_sum']. shift(1) * 0.5,  # Simple snowmelt approximation
        0
    )
    
    # Recharge proxy
    df['recharge_proxy'] = (
        df['total_precipitation_sum'] + 
        df['potential_snowmelt'] - 
        np.abs(df['evaporation_sum'])
    )
    
    # Aridity index proxy (ET / Precipitation) - avoid division by zero
    df['aridity_index'] = np. where (
        df['total_precipitation_sum'] > 0.0001,
        np.abs(df['evaporation_sum']) / df['total_precipitation_sum'],
        np.abs(df['evaporation_sum']) / 0.0001
    )
    df['aridity_index'] = df['aridity_index'].clip(upper=100)  # Cap extreme values
    
    # Temperature-precipitation interaction
    df['temp_precip_interaction'] = df['temp_celsius_mean'] * df['total_precipitation_sum']
    
    # Diurnal temperature range
    df['diurnal_temp_range'] = df['temp_celsius_max'] - df['temp_celsius_min']
    
    # Growing degree days (base 10°C)
    df['growing_degree_days'] = np.maximum(0, df['temp_celsius_mean'] - 10)
    
    # Heating degree days (base 18°C)
    df['heating_degree_days'] = np.maximum(0, 18 - df['temp_celsius_mean'])
    
    # Cooling degree days (base 18°C)
    df['cooling_degree_days'] = np.maximum(0, df['temp_celsius_mean'] - 18)
    
    # Net radiation energy available for evaporation
    df['net_energy_index'] = df['solar_radiation_MJ'] * (1 - df['total_cloud_cover_mean'])
    
    # Wind chill / Heat stress index approximation
    df['wind_temp_index'] = df['temp_celsius_mean'] - (0.5 * df['wind_speed'])
    
    print(f"✓ Added 12 hydrological/derived features")
    
    return df


# =============================================================================
# 7. RATE OF CHANGE FEATURES
# =============================================================================
def add_rate_of_change_features(df:  pd.DataFrame,
                                 target_col: str = 'water_level') -> pd.DataFrame:
    """
    Add rate of change (derivative) features to capture trends.
    """
    df = df.copy()

    # All target-derived rate-of-change features are computed on the lag-1
    # series (target_shifted[t] == target_col[t-1]) so that, at row t, they
    # only ever encode information available through t-1. Computing these
    # directly on df[target_col] would leak the current-day/target value
    # (e.g. diff_1d = target[t] - target[t-1] contains target[t] itself).
    target_shifted = df[target_col].shift(1)

    # Daily change in water level (as of t-1)
    df[f'{target_col}_diff_1d'] = target_shifted.diff(1)
    df[f'{target_col}_diff_7d'] = target_shifted.diff(7)
    df[f'{target_col}_diff_30d'] = target_shifted.diff(30)

    # Percentage change (as of t-1)
    df[f'{target_col}_pct_change_1d'] = target_shifted.pct_change(1)
    df[f'{target_col}_pct_change_7d'] = target_shifted.pct_change(7)
    df[f'{target_col}_pct_change_30d'] = target_shifted.pct_change(30)

    # Acceleration (second derivative, as of t-1)
    df[f'{target_col}_acceleration'] = df[f'{target_col}_diff_1d'].diff(1)

    # Climate rate of change
    climate_cols = ['temp_celsius_mean', 'total_precipitation_sum', 'evaporation_sum']
    for col in climate_cols:
        df[f'{col}_diff_1d'] = df[col].diff(1)
        df[f'{col}_diff_7d'] = df[col].diff(7)

    # Trend indicator (comparing t-1 to its trailing rolling mean)
    df[f'{target_col}_vs_30d_mean'] = (
        target_shifted - target_shifted.rolling(30, min_periods=1).mean()
    )
    df[f'{target_col}_vs_90d_mean'] = (
        target_shifted - target_shifted.rolling(90, min_periods=1).mean()
    )
    
    print(f"✓ Added 15 rate of change features")
    
    return df


# =============================================================================
# 8. INTERACTION FEATURES
# =============================================================================
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between key variables.
    """
    df = df.copy()
    
    # Temperature and radiation interaction
    df['temp_radiation_interaction'] = df['temp_celsius_mean'] * df['solar_radiation_MJ']
    
    # Wind and evaporation interaction
    df['wind_evap_interaction'] = df['wind_speed'] * np.abs(df['evaporation_sum'])
    
    # Cloud cover and radiation interaction
    df['cloud_radiation_interaction'] = df['total_cloud_cover_mean'] * df['solar_radiation_MJ']
    
    # Pressure and temperature interaction (related to weather systems)
    df['pressure_temp_interaction'] = (
        df['surface_pressure_mean'] / 1000  # Normalize pressure
    ) * df['temp_celsius_mean']
    
    # Soil temperature and air temperature difference
    df['soil_air_temp_diff'] = df['soil_temp_celsius'] - df['temp_celsius_mean']
    
    # Precipitation intensity (precip / cloud cover) - when cloudy
    df['precip_intensity'] = np.where(
        df['total_cloud_cover_mean'] > 0.1,
        df['total_precipitation_sum'] / df['total_cloud_cover_mean'],
        0
    )
    
    print(f"✓ Added 6 interaction features")
    
    return df


# =============================================================================
# 9. ANOMALY FEATURES
# =============================================================================
def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that identify anomalies or deviations from normal. 
    """
    df = df.copy()
    
    # Calculate monthly climatology (normal values)
    monthly_stats = df.groupby('month').agg({
        'temp_celsius_mean':  ['mean', 'std'],
        'total_precipitation_sum': ['mean', 'std'],
        'water_level': ['mean', 'std']
    })
    monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]
    monthly_stats = monthly_stats.reset_index()
    
    # Merge climatology back
    df = df.merge(monthly_stats, on='month', how='left', suffixes=('', '_monthly'))
    
    # Temperature anomaly (z-score)
    df['temp_anomaly'] = (
        (df['temp_celsius_mean'] - df['temp_celsius_mean_mean']) / 
        df['temp_celsius_mean_std']. replace(0, 1)
    )
    
    # Precipitation anomaly
    df['precip_anomaly'] = (
        (df['total_precipitation_sum'] - df['total_precipitation_sum_mean']) / 
        df['total_precipitation_sum_std'].replace(0, 1)
    )
    
    # Water level anomaly - computed on the lag-1 value (t-1 vs. the t-1
    # month's climatology), not the current-day value, since
    # (water_level[t] - mean)/std is a near-affine transform of the target
    # itself and would leak it directly if used as a same-row feature.
    water_level_lag1 = df['water_level'].shift(1)
    df['water_level_anomaly'] = (
        (water_level_lag1 - df['water_level_mean']) /
        df['water_level_std']. replace(0, 1)
    )

    # Binary flags for extreme values
    df['is_hot_day'] = (df['temp_anomaly'] > 2).astype(int)
    df['is_cold_day'] = (df['temp_anomaly'] < -2).astype(int)
    df['is_wet_day'] = (df['precip_anomaly'] > 2).astype(int)
    df['is_dry_day'] = (df['precip_anomaly'] < -1).astype(int)
    
    # Drop the temporary columns used for calculation
    cols_to_drop = [
        'temp_celsius_mean_mean', 'temp_celsius_mean_std',
        'total_precipitation_sum_mean', 'total_precipitation_sum_std',
        'water_level_mean', 'water_level_std'
    ]
    df = df.drop(columns=cols_to_drop)
    
    print(f"✓ Added 7 anomaly features")
    
    return df


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================
def run_feature_engineering_pipeline(filepath: str, 
                                      output_path: str = None) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline. 
    """
    print("="*70)
    print("         FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Load data
    print("\n[1/10] Loading data...")
    df = load_and_prepare_data(filepath)
    initial_cols = len(df.columns)
    
    # Apply feature engineering steps
    print("\n[2/10] Adding temporal features...")
    df = add_temporal_features(df)
    
    print("\n[3/10] Adding cyclical features...")
    df = add_cyclical_features(df)
    
    print("\n[4/10] Adding lag features...")
    df = add_lag_features(df)
    
    print("\n[5/10] Adding rolling statistics...")
    df = add_rolling_features(df)
    
    print("\n[6/10] Adding cumulative features...")
    df = add_cumulative_features(df)
    
    print("\n[7/10] Adding hydrological features...")
    df = add_hydrological_features(df)
    
    print("\n[8/10] Adding rate of change features...")
    df = add_rate_of_change_features(df)
    
    print("\n[9/10] Adding interaction features...")
    df = add_interaction_features(df)
    
    print("\n[10/10] Adding anomaly features...")
    df = add_anomaly_features(df)
    
    # Summary
    final_cols = len(df.columns)
    new_features = final_cols - initial_cols
    
    print("\n" + "="*70)
    print("                    PIPELINE COMPLETE")
    print("="*70)
    print(f"Initial features:   {initial_cols}")
    print(f"Final features:    {final_cols}")
    print(f"New features added: {new_features}")
    print(f"Total records:     {len(df)}")
    
    # Handle missing values created by lag/diff operations
    print("\n[INFO] Handling missing values...")
    missing_before = df.isnull().sum().sum()
    
    # Option 1: Drop rows with NaN (loses initial rows due to lags)
    # df_clean = df.dropna()
    
    # Option 2: Forward fill then backward fill (keeps all rows)
    # df_clean = df.fillna(method='ffill').fillna(method='bfill')
    
    # Option 3: Keep NaN and handle during model training (recommended for TFT)
    df_clean = df. copy()
    
    print(f"Missing values:  {missing_before}")
    print(f"Note: Missing values from lag features will be handled during train/test split")
    
    # Save engineered dataset
    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"\n✓ Saved engineered dataset to: {output_path}")
    
    return df_clean


# =============================================================================
# FEATURE CATEGORIZATION (for TFT model)
# =============================================================================
def categorize_features_for_tft(df: pd.DataFrame) -> dict:
    """
    Categorize features for Temporal Fusion Transformer. 
    TFT requires explicit categorization of feature types.
    """
    
    # Target variable
    target = 'water_level'
    
    # Time index
    time_idx = 'date'
    
    # Static categorical features (don't change over time for this location)
    # In single-well study, we might not have many static features
    static_categoricals = []
    
    # Static real-valued features
    static_reals = []
    
    # Time-varying known features (known in advance)
    time_varying_known_categoricals = [
        'month', 'quarter', 'season', 'day_of_week', 'is_weekend'
    ]
    
    time_varying_known_reals = [
        'day_of_year', 'week_of_year', 'days_since_start',
        'day_of_year_sin', 'day_of_year_cos',
        'month_sin', 'month_cos',
        'week_sin', 'week_cos',
        'day_of_week_sin', 'day_of_week_cos'
    ]
    
    # Time-varying unknown features (only known up to current time)
    time_varying_unknown_reals = [
        # Original climate variables
        'temp_celsius_mean', 'temp_celsius_min', 'temp_celsius_max',
        'wind_speed', 'surface_pressure_mean',
        'total_precipitation_sum', 'solar_radiation_MJ',
        'total_cloud_cover_mean', 'evaporation_sum',
        'soil_temperature_level_1_mean', 'snowfall_sum',
        
        # Derived features
        'water_balance', 'recharge_proxy', 'aridity_index',
        'diurnal_temp_range', 'growing_degree_days',
        'net_energy_index', 'soil_air_temp_diff',
        
        # Lag features (subset - most important)
        'water_level_lag_1d', 'water_level_lag_7d', 'water_level_lag_30d',
        
        # Rolling features (subset)
        'water_level_rolling_mean_7d', 'water_level_rolling_mean_30d',
        'total_precipitation_sum_rolling_sum_7d',
        'total_precipitation_sum_rolling_sum_30d',
        
        # Rate of change
        'water_level_diff_1d', 'water_level_diff_7d',
        
        # Anomalies
        'temp_anomaly', 'precip_anomaly'
    ]
    
    feature_config = {
        'target': target,
        'time_idx':  time_idx,
        'static_categoricals': static_categoricals,
        'static_reals':  static_reals,
        'time_varying_known_categoricals': time_varying_known_categoricals,
        'time_varying_known_reals': time_varying_known_reals,
        'time_varying_unknown_reals': time_varying_unknown_reals
    }
    
    print("\n" + "="*70)
    print("         FEATURE CATEGORIZATION FOR TFT")
    print("="*70)
    print(f"Target:  {target}")
    print(f"Time-varying known categoricals: {len(time_varying_known_categoricals)}")
    print(f"Time-varying known reals: {len(time_varying_known_reals)}")
    print(f"Time-varying unknown reals:  {len(time_varying_unknown_reals)}")
    
    return feature_config


# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # Run pipeline
    df_engineered = run_feature_engineering_pipeline(
        filepath='data/merged/merged_dataset.csv',
        output_path='data/engineered/engineered_dataset.csv'
    )
    
    # Get feature categorization for TFT
    feature_config = categorize_features_for_tft(df_engineered)
    
    # Display sample of engineered features
    print("\n" + "="*70)
    print("         SAMPLE OF ENGINEERED DATASET")
    print("="*70)
    print(df_engineered.head())
    
    # Display all column names
    print("\n" + "="*70)
    print("         ALL FEATURES")
    print("="*70)
    for i, col in enumerate(df_engineered. columns, 1):
        print(f"{i: 3}. {col}")