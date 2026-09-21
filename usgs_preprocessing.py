import pandas as pd
import numpy as np

def preprocess_usgs(filepath):
    """
    Preprocess USGS groundwater level data. 
    """
    # Load raw data
    df = pd. read_csv(filepath, parse_dates=['datetime'])
    
    # Extract date
    df['date'] = df['datetime'].dt. date
    
    # Handle multiple readings per day (if any)
    df_daily = df.groupby('date').agg({
        'water_level': 'mean'  # feet below land surface
    }).reset_index()
    
    # Quality control
    # Remove obvious outliers (adjust thresholds based on your data)
    q1 = df_daily['water_level']. quantile(0.01)
    q99 = df_daily['water_level']. quantile(0.99)
    df_daily = df_daily[
        (df_daily['water_level'] >= q1) & 
        (df_daily['water_level'] <= q99)
    ]
    
    # Check for gaps
    date_range = pd. date_range(
        start=df_daily['date'].min(), 
        end=df_daily['date'].max(), 
        freq='D'
    )
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily. set_index('date').reindex(date_range)
    df_daily. index. name = 'date'
    
    # Flag missing values (don't interpolate yet - document gaps)
    df_daily['is_interpolated'] = df_daily['water_level'].isna()
    
    # Interpolate small gaps (e.g., ≤ 7 days)
    df_daily['water_level'] = df_daily['water_level'].interpolate(
        method='linear', 
        limit=7
    )
    
    return df_daily. reset_index()

# Execute
usgs_clean = preprocess_usgs('data/raw/usgs_gwl_raw.csv')
usgs_clean.to_csv('data/processed/usgs_gwl_clean.csv', index=False)