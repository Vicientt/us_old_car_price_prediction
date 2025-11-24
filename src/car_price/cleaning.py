import pandas as pd
import numpy as np
from src.car_price import config

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean raw data.
    """
    df = df.copy()

    # --- RENAME COLUMNS ---
    rename_dict = {k: v for k, v in config.COLUMN_MAPPING.items() if k in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # 1. Drop unnecessary columns
    cols_to_drop = [c for c in config.DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # 2. Handle Missing Values
    for col in config.CAT_COLS_FILL_OTHER:
        if col not in df.columns:
            df[col] = 'other'
        else:
            df[col] = df[col].fillna('other')   
        df[col] = df[col].astype(str)
            
    # 3. Handle YEAR and ODOMETER (CRITICAL FIX)
    if 'year' in df.columns:
        df['year'] = df['year'].fillna(2011)
        
        # --- FIXED HERE ---
        # The dataset is from approx 2021. 
        # If we use 2025, the car becomes 4 years older, causing the price to drop.
        dataset_year_anchor = 2021 
        df['year_to_now'] = dataset_year_anchor - df['year']
    else:
        # Default age if year is missing
        df['year_to_now'] = 10.0 
        
    if 'odometer' in df.columns:
        df['odometer'] = df['odometer'].fillna(90000)
    else:
        df['odometer'] = 90000.0
        
    # 4. Clean text
    if 'model' in df.columns:
        df['model'] = df['model'].str.strip().str.lower()
        
    if 'manufacturer' in df.columns:
        df['manufacturer'] = df['manufacturer'].str.strip().str.lower()

    return df