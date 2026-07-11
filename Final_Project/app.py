"""
Store Item Demand Forecasting — Web Dashboard
===============================================
Flask-based interactive dashboard for exploring demand forecasting results.

Endpoints:
    /                  : Main dashboard page
    /api/summary       : Model performance summary & statistics
    /api/predictions   : Paginated predictions data
    /api/train_stats   : Training data aggregated statistics
    /api/store_item    : Store-item level sales data
    /api/timeseries    : Daily time series for a given store-item pair

Usage:
    python app.py
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------
_cache = {}

def get_train():
    if 'train' not in _cache:
        _cache['train'] = pd.read_csv(
            os.path.join('data', 'train.csv'), parse_dates=['date']
        )
    return _cache['train']

def get_submission():
    if 'submission' not in _cache:
        _cache['submission'] = pd.read_csv(os.path.join('output', 'submission.csv'))
    return _cache['submission']

def get_test():
    if 'test' not in _cache:
        _cache['test'] = pd.read_csv(os.path.join('data', 'test.csv'))
        _cache['test']['date'] = pd.to_datetime(_cache['test']['date'], format='%d-%m-%y')
    return _cache['test']


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/summary')
def api_summary():
    """Return model performance summary and dataset statistics."""
    train = get_train()
    sub = get_submission()

    summary = {
        'model': {
            'validation_smape': 12.3238,
            'best_iteration': 1633,
            'final_iterations': 1796,
            'num_features': 35,
            'algorithm': 'LightGBM',
            'objective': 'regression_l1 (MAE on log1p sales)',
        },
        'dataset': {
            'train_rows': len(train),
            'test_rows': len(sub),
            'num_stores': int(train['store'].nunique()),
            'num_items': int(train['item'].nunique()),
            'date_range_train': f"{train['date'].min().strftime('%Y-%m-%d')} to {train['date'].max().strftime('%Y-%m-%d')}",
            'date_range_test': '2018-01-01 to 2018-03-31',
        },
        'train_stats': {
            'mean': round(float(train['sales'].mean()), 2),
            'std': round(float(train['sales'].std()), 2),
            'min': int(train['sales'].min()),
            'max': int(train['sales'].max()),
            'median': round(float(train['sales'].median()), 2),
        },
        'prediction_stats': {
            'mean': round(float(sub['sales'].mean()), 2),
            'std': round(float(sub['sales'].std()), 2),
            'min': round(float(sub['sales'].min()), 2),
            'max': round(float(sub['sales'].max()), 2),
            'median': round(float(sub['sales'].median()), 2),
        },
        'feature_importance': [
            {'feature': 'item', 'importance': 6168, 'category': 'identifier'},
            {'feature': 'sales_lag_365', 'importance': 4528, 'category': 'lag'},
            {'feature': 'month', 'importance': 3418, 'category': 'calendar'},
            {'feature': 'store_item_mean', 'importance': 2947, 'category': 'encoding'},
            {'feature': 'sales_roll_mean_90_365', 'importance': 2818, 'category': 'rolling'},
            {'feature': 'dayofweek', 'importance': 2516, 'category': 'calendar'},
            {'feature': 'sales_roll_std_90_365', 'importance': 2082, 'category': 'rolling'},
            {'feature': 'sales_lag_91', 'importance': 1940, 'category': 'lag'},
            {'feature': 'sales_lag_98', 'importance': 1393, 'category': 'lag'},
            {'feature': 'sales_roll_std_90_180', 'importance': 1270, 'category': 'rolling'},
            {'feature': 'sales_lag_119', 'importance': 1256, 'category': 'lag'},
            {'feature': 'store_item_std', 'importance': 1173, 'category': 'encoding'},
            {'feature': 'dayofyear', 'importance': 1144, 'category': 'calendar'},
            {'feature': 'sales_roll_std_90_7', 'importance': 1144, 'category': 'rolling'},
            {'feature': 'sales_roll_std_90_30', 'importance': 1114, 'category': 'rolling'},
        ]
    }
    return jsonify(summary)


@app.route('/api/predictions')
def api_predictions():
    """Return predictions merged with test metadata, with optional filters."""
    sub = get_submission()
    test = get_test()

    merged = test.merge(sub, on='id')
    
    # Optional filters
    store = request.args.get('store', type=int)
    item = request.args.get('item', type=int)
    if store:
        merged = merged[merged['store'] == store]
    if item:
        merged = merged[merged['item'] == item]

    merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
    
    return jsonify({
        'count': len(merged),
        'data': merged.to_dict(orient='records')
    })


@app.route('/api/train_stats')
def api_train_stats():
    """Return aggregated training statistics for charts."""
    train = get_train()

    # Monthly average sales across all years
    train['month'] = train['date'].dt.month
    train['year'] = train['date'].dt.year
    monthly = train.groupby('month')['sales'].mean().reset_index()
    monthly.columns = ['month', 'avg_sales']

    # Monthly by year
    monthly_by_year = train.groupby(['year', 'month'])['sales'].mean().reset_index()
    monthly_by_year.columns = ['year', 'month', 'avg_sales']

    # Day of week
    train['dayofweek'] = train['date'].dt.dayofweek
    weekly = train.groupby('dayofweek')['sales'].mean().reset_index()
    weekly.columns = ['dayofweek', 'avg_sales']
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly['day_name'] = weekly['dayofweek'].map(lambda x: day_names[x])

    # Store-level
    store_avg = train.groupby('store')['sales'].mean().reset_index()
    store_avg.columns = ['store', 'avg_sales']

    return jsonify({
        'monthly': monthly.to_dict(orient='records'),
        'monthly_by_year': monthly_by_year.to_dict(orient='records'),
        'weekly': weekly.to_dict(orient='records'),
        'store_avg': store_avg.round(2).to_dict(orient='records'),
    })


@app.route('/api/store_item')
def api_store_item():
    """Return store-item heatmap data."""
    train = get_train()
    pivot = train.groupby(['store', 'item'])['sales'].mean().reset_index()
    pivot.columns = ['store', 'item', 'avg_sales']
    pivot['avg_sales'] = pivot['avg_sales'].round(2)
    return jsonify(pivot.to_dict(orient='records'))


@app.route('/api/timeseries')
def api_timeseries():
    """Return daily time series for a given store-item pair, plus predictions."""
    store = request.args.get('store', 1, type=int)
    item = request.args.get('item', 1, type=int)

    train = get_train()
    subset = train[(train['store'] == store) & (train['item'] == item)].sort_values('date')

    # Add predictions
    test = get_test()
    sub = get_submission()
    test_subset = test[(test['store'] == store) & (test['item'] == item)].merge(sub, on='id')
    test_subset = test_subset.sort_values('date')

    result = {
        'store': store,
        'item': item,
        'history': {
            'dates': subset['date'].dt.strftime('%Y-%m-%d').tolist(),
            'sales': subset['sales'].tolist(),
        },
        'forecast': {
            'dates': test_subset['date'].dt.strftime('%Y-%m-%d').tolist(),
            'sales': test_subset['sales'].round(2).tolist(),
        }
    }
    return jsonify(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  Store Item Demand Forecasting Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, port=5000)
