import pandas as pd
import numpy as np

def preprocess_era5(filepath):
    """
    Preprocess ERA5 6-hourly data to daily resolution. 
    """
    # Load raw data
    df = pd.read_csv(filepath, parse_dates=['time'])
    
    # Extract date from datetime
    df['date'] = df['time'].dt. date
    
    # Define aggregation rules
    agg_rules = {
        '2m_temperature':  ['mean', 'min', 'max'],
        '10m_u_component_of_wind':  'mean',
        '10m_v_component_of_wind': 'mean',
        'surface_pressure': 'mean',
        'total_precipitation': 'sum',
        'surface_net_solar_radiation': 'sum',
        'total_cloud_cover': 'mean',
        'evaporation': 'sum',
        'soil_temperature_level_1': 'mean',
        'snowfall': 'sum'
    }
    
    # Aggregate to daily
    df_daily = df.groupby('date').agg(agg_rules)
    
    # Flatten column names
    df_daily.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                        for col in df_daily.columns]
    
    df_daily = df_daily.reset_index()
    
    # Calculate derived variables
    # Wind speed from u and v components
    df_daily['wind_speed'] = np.sqrt(
        df_daily['10m_u_component_of_wind_mean']**2 + 
        df_daily['10m_v_component_of_wind_mean']**2
    )
    
    # Temperature in Celsius (if in Kelvin)
    df_daily['temp_celsius_mean'] = df_daily['2m_temperature_mean'] - 273.15
    df_daily['temp_celsius_min'] = df_daily['2m_temperature_min'] - 273.15
    df_daily['temp_celsius_max'] = df_daily['2m_temperature_max'] - 273.15
    
    # Convert radiation from J/m² to MJ/m²/day
    df_daily['solar_radiation_MJ'] = df_daily['surface_net_solar_radiation_sum'] / 1e6
    
    return df_daily

# Execute
era5_daily = preprocess_era5('data/raw/era5_raw.csv')
era5_daily.to_csv('data/processed/era5_daily.csv', index=False)