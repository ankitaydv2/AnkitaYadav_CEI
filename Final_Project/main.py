"""
Store Item Demand Forecasting — Main Pipeline
================================================
End-to-end orchestrator for the Kaggle Store Item Demand Forecasting Challenge.
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview

Pipeline Steps:
    1. Load & parse train.csv (YYYY-MM-DD) and test.csv (DD-MM-YY)
    2. Engineer 35 features (calendar, lag, rolling, target encodings)
    3. Train LightGBM with time-based validation (SMAPE evaluation)
    4. Retrain on full dataset and generate 90-day test forecasts
    5. Save submission.csv to output/

Usage:
    python main.py
"""

import os
import pandas as pd
import numpy as np
from src.features import create_features, add_target_encodings
from src.model import train_and_validate, train_final_and_predict

def main():
    print("Starting Store Item Demand Forecasting Pipeline...")
    
    # 1. Load Data
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Data files not found. Running download_data.py...")
        import subprocess
        subprocess.run(["py", "download_data.py"], check=True)
        
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], format='%Y-%m-%d')
    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], format='%d-%m-%y')
    
    print(f"Loaded Train set: {train.shape} (from {train['date'].min()} to {train['date'].max()})")
    print(f"Loaded Test set: {test.shape} (from {test['date'].min()} to {test['date'].max()})")
    
    # Combine datasets for consistent feature engineering
    # test dataset has columns: id, date, store, item
    # train dataset has columns: date, store, item, sales
    df = pd.concat([train, test], ignore_index=True)
    
    # 2. Feature Engineering (Lags and Rolling Window stats)
    df = create_features(df)
    
    # Define initial lists of features and categorical variables
    drop_cols = ['date', 'sales', 'log_sales', 'id']
    categorical_features = ['store', 'item', 'month', 'dayofweek']
    
    # 3. Model Validation
    print("\n--- Phase 1: Model Validation ---")
    # To prevent validation leakage, calculate target encodings ONLY on validation train split (< 2017-10-01)
    val_train_mask = (df['date'] < '2017-10-01') & (df['sales'].notna())
    df_val = add_target_encodings(df.copy(), val_train_mask)
    
    # Get all features including target encoding columns
    val_features = [c for c in df_val.columns if c not in drop_cols]
    print(f"Features used for training: {val_features}")
    
    # Train validation model and get best iteration
    best_iteration, val_smape = train_and_validate(df_val, val_features, categorical_features)
    
    # 4. Final Retraining & Prediction
    print("\n--- Phase 2: Final Retraining & Inference ---")
    # For final predictions, calculate target encodings on full train dataset
    full_train_mask = df['sales'].notna()
    df_final = add_target_encodings(df.copy(), full_train_mask)
    
    final_features = [c for c in df_final.columns if c not in drop_cols]
    
    # Train final model on full training set and predict on test set
    sub_df = train_final_and_predict(df_final, final_features, categorical_features, best_iteration)
    
    # Save predictions
    os.makedirs("output", exist_ok=True)
    sub_path = os.path.join("output", "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    
    print("\n--- Pipeline Completed ---")
    print(f"Submission saved to {sub_path} successfully. Shape: {sub_df.shape}")
    print("First 10 rows of submission:")
    print(sub_df.head(10))

if __name__ == "__main__":
    main()
