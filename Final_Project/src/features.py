"""
Feature Engineering Pipeline for Store Item Demand Forecasting
===============================================================
Generates 35 features across 4 categories:
  1. Calendar Features  (7) : year, month, day, dayofweek, dayofyear, is_weekend, weekofyear
  2. Lag Features       (10): Grouped lags at 90–365 day horizons on log-transformed sales
  3. Rolling Features   (10): Rolling mean & std of lag_90 over 7/30/90/180/365-day windows
  4. Target Encodings   (6) : Store, item, and store-item level mean & std of log sales

Key Design Decision:
    Minimum lag = 90 days, matching the 90-day test prediction horizon.
    This prevents data leakage in a direct (non-recursive) forecasting setup.
"""

import pandas as pd
import numpy as np

def create_features(df):
    """
    Generate date-time, lag, and rolling window features.
    The input DataFrame should be a combined train and test DataFrame.
    """
    print("Generating date-time features...")
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort dataset by store, item, and date to ensure correct lag/rolling calculations
    df = df.sort_values(by=['store', 'item', 'date']).reset_index(drop=True)
    
    # 1. Calendar/Date Features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    # 2. Target Transformation
    df['log_sales'] = np.log1p(df['sales'])
    
    # 3. Lag Features
    # Since the forecasting horizon is 90 days, the minimum lag is 90 days
    lags = [90, 91, 92, 98, 105, 112, 119, 126, 180, 365]
    print(f"Generating lag features: {lags}...")
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['log_sales'].shift(lag)
        
    # 4. Rolling Window Features based on lag 90
    windows = [7, 30, 90, 180, 365]
    print(f"Generating rolling window features for windows: {windows}...")
    for w in windows:
        # Rolling mean of lag_90
        df[f'sales_roll_mean_90_{w}'] = df.groupby(['store', 'item'])['sales_lag_90'].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        # Rolling standard deviation of lag_90
        df[f'sales_roll_std_90_{w}'] = df.groupby(['store', 'item'])['sales_lag_90'].transform(
            lambda x: x.rolling(w, min_periods=1).std()
        )
        
    return df

def add_target_encodings(df, train_mask):
    """
    Calculate and add target encodings (mean, std of sales) based on the training split only
    to avoid data leakage into validation or test sets.
    """
    print("Calculating target encodings on training split...")
    train_df = df[train_mask]
    
    # Store-Item level statistics
    store_item_stats = train_df.groupby(['store', 'item'])['log_sales'].agg(['mean', 'std']).reset_index()
    store_item_stats.rename(columns={'mean': 'store_item_mean', 'std': 'store_item_std'}, inplace=True)
    
    # Item level statistics
    item_stats = train_df.groupby('item')['log_sales'].agg(['mean', 'std']).reset_index()
    item_stats.rename(columns={'mean': 'item_mean', 'std': 'item_std'}, inplace=True)
    
    # Store level statistics
    store_stats = train_df.groupby('store')['log_sales'].agg(['mean', 'std']).reset_index()
    store_stats.rename(columns={'mean': 'store_mean', 'std': 'store_std'}, inplace=True)
    
    # Merge back to the main dataframe
    df = df.merge(store_item_stats, on=['store', 'item'], how='left')
    df = df.merge(item_stats, on='item', how='left')
    df = df.merge(store_stats, on='store', how='left')
    
    return df
