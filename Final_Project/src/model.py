"""
LightGBM Model Training & Inference Pipeline
=============================================
Implements a two-phase forecasting workflow:

Phase 1 - Validation:
    Train on data before 2017-10-01, validate on Oct–Dec 2017.
    Uses early stopping (150 rounds) to find optimal iteration count.
    Reports SMAPE on original-scale sales values.

Phase 2 - Final Inference:
    Retrain on the full 5-year training set using 1.1× the best iteration count.
    Generate predictions for the 90-day test horizon (Jan–Mar 2018).

Model Configuration:
    - Objective: regression_l1 (MAE on log1p-transformed sales)
    - This serves as an excellent proxy for SMAPE optimization.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred)
    # Avoid division by zero
    mask = denominator != 0
    return 200 * np.mean(diff[mask] / denominator[mask])

def train_and_validate(df, features, categorical_features):
    """
    Train LightGBM on train data (< 2017-10-01) and evaluate on validation data (last 3 months of 2017).
    """
    # Create masks
    train_mask = (df['date'] < '2017-10-01') & (df['sales'].notna())
    val_mask = (df['date'] >= '2017-10-01') & (df['date'] <= '2017-12-31') & (df['sales'].notna())
    
    # Split
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, 'log_sales']
    X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, 'log_sales']
    
    # Original target values for validation evaluation
    y_val_orig = df.loc[val_mask, 'sales'].values
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Set parameters
    params = {
        'n_estimators': 3000,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'regression_l1',  # L1 loss on log(1 + sales) is a great proxy for SMAPE
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=True),
            lgb.log_evaluation(period=100)
        ],
        categorical_feature=categorical_features
    )
    
    # Predict on validation set
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    
    # Calculate SMAPE
    val_smape = smape(y_val_orig, val_preds)
    print(f"\n---> Validation SMAPE: {val_smape:.4f}% <---\n")
    
    # Print feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print("Top 15 Features by Importance:")
    print(importance.head(15).to_string(index=False))
    
    best_iteration = model.best_iteration_
    if best_iteration is None or best_iteration == 0:
        best_iteration = 1000
    
    return best_iteration, val_smape

def train_final_and_predict(df, features, categorical_features, best_iteration):
    """
    Retrain the model on the full training set and forecast the test set sales.
    """
    train_mask = df['sales'].notna()
    test_mask = df['id'].notna()
    
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, 'log_sales']
    X_test = df.loc[test_mask, features]
    
    # Increase iterations slightly because we train on 100% of train data instead of ~95%
    final_iterations = int(best_iteration * 1.1)
    print(f"Training final model on full training set with {final_iterations} iterations...")
    
    params = {
        'n_estimators': final_iterations,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'regression_l1',
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        categorical_feature=categorical_features
    )
    
    print("Predicting sales for the test set...")
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    
    # Create final submission dataframe
    sub_df = pd.DataFrame({
        'id': df.loc[test_mask, 'id'].astype(int),
        'sales': test_preds
    }).sort_values(by='id').reset_index(drop=True)
    
    return sub_df
