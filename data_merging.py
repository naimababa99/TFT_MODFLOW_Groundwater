import pandas as pd

def merge_datasets(era5_path, usgs_path):
    """
    Merge preprocessed ERA5 and USGS datasets.
    """
    # Load preprocessed data
    era5 = pd.read_csv(era5_path, parse_dates=['date'])
    usgs = pd.read_csv(usgs_path, parse_dates=['date'])
    
    # Merge on date
    merged = pd.merge(
        era5, 
        usgs, 
        on='date', 
        how='inner'  # Only keep dates present in both datasets
    )
    
    # Sort by date
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Report merge statistics
    print(f"ERA5 records: {len(era5)}")
    print(f"USGS records: {len(usgs)}")
    print(f"Merged records: {len(merged)}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"Missing values:\n{merged.isnull().sum()}")
    
    return merged

# Execute
merged_df = merge_datasets(
    'data/processed/era5_daily.csv',
    'data/processed/usgs_gwl_clean.csv'
)
merged_df.to_csv('data/merged/merged_dataset.csv', index=False)