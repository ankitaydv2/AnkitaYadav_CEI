"""
Exploratory Data Analysis & Visualization Script
=================================================
Generates publication-quality plots for the Store Item Demand Forecasting project.

Outputs saved to: output/plots/
    - feature_importance.png    : Top 15 LightGBM features by split count
    - prediction_distribution.png : Histogram of forecasted sales values
    - monthly_sales_trend.png   : Average daily sales by month (seasonality)
    - store_item_heatmap.png    : Mean sales heatmap across stores and items

Usage:
    python eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PLOTS_DIR = os.path.join("output", "plots")
TRAIN_PATH = os.path.join("data", "train.csv")
SUBMISSION_PATH = os.path.join("output", "submission.csv")

# Publication-quality style settings
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Consistent color palette
COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'accent': '#059669',
    'warm': '#dc2626',
    'gradient': plt.cm.viridis,
}


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data():
    """Load training data and submission file."""
    train = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
    
    submission = None
    if os.path.exists(SUBMISSION_PATH):
        submission = pd.read_csv(SUBMISSION_PATH)
    
    return train, submission


# ---------------------------------------------------------------------------
# Plot 1: Feature Importance (from trained model results)
# ---------------------------------------------------------------------------
def plot_feature_importance():
    """
    Plot the top 15 features by LightGBM split importance.
    Uses pre-computed results from the model training output.
    """
    # Feature importance data from the model training run
    features_data = {
        'feature': [
            'item', 'sales_lag_365', 'month', 'store_item_mean',
            'sales_roll_mean_90_365', 'dayofweek', 'sales_roll_std_90_365',
            'sales_lag_91', 'sales_lag_98', 'sales_roll_std_90_180',
            'sales_lag_119', 'store_item_std', 'dayofyear',
            'sales_roll_std_90_7', 'sales_roll_std_90_30'
        ],
        'importance': [
            6168, 4528, 3418, 2947, 2818, 2516, 2082,
            1940, 1393, 1270, 1256, 1173, 1144, 1144, 1114
        ]
    }
    
    df_imp = pd.DataFrame(features_data)
    
    # Color bars by feature category
    category_colors = []
    for f in df_imp['feature']:
        if 'lag' in f:
            category_colors.append('#2563eb')      # Blue for lags
        elif 'roll' in f:
            category_colors.append('#7c3aed')      # Purple for rolling
        elif f in ('store_item_mean', 'store_item_std', 'item_mean', 'store_mean'):
            category_colors.append('#059669')      # Green for target encoding
        else:
            category_colors.append('#dc2626')      # Red for calendar/identifiers
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(
        df_imp['feature'][::-1],
        df_imp['importance'][::-1],
        color=category_colors[::-1],
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.set_xlabel('Number of Splits')
    ax.set_title('Top 15 Features by LightGBM Split Importance', fontweight='bold', pad=15)
    
    # Add value labels on bars
    for bar, val in zip(bars, df_imp['importance'][::-1]):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=9, color='#374151')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#dc2626', label='Calendar / Identifier'),
        Patch(facecolor='#2563eb', label='Lag Features'),
        Patch(facecolor='#7c3aed', label='Rolling Window'),
        Patch(facecolor='#059669', label='Target Encoding'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Prediction Distribution
# ---------------------------------------------------------------------------
def plot_prediction_distribution(submission):
    """
    Histogram comparing the distribution of predicted sales
    against historical training sales.
    """
    train = pd.read_csv(TRAIN_PATH)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training sales distribution
    axes[0].hist(train['sales'], bins=80, color=COLORS['primary'],
                 alpha=0.8, edgecolor='white', linewidth=0.3)
    axes[0].set_title('Training Data: Sales Distribution', fontweight='bold')
    axes[0].set_xlabel('Daily Sales')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(train['sales'].mean(), color=COLORS['warm'],
                    linestyle='--', linewidth=1.5, label=f"Mean: {train['sales'].mean():.1f}")
    axes[0].legend()
    
    # Predicted sales distribution
    axes[1].hist(submission['sales'], bins=80, color=COLORS['secondary'],
                 alpha=0.8, edgecolor='white', linewidth=0.3)
    axes[1].set_title('Predicted: Sales Distribution', fontweight='bold')
    axes[1].set_xlabel('Predicted Daily Sales')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(submission['sales'].mean(), color=COLORS['warm'],
                    linestyle='--', linewidth=1.5, label=f"Mean: {submission['sales'].mean():.1f}")
    axes[1].legend()
    
    plt.suptitle('Sales Distribution: Training vs Predicted', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'prediction_distribution.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Monthly Sales Trend (Seasonality)
# ---------------------------------------------------------------------------
def plot_monthly_sales_trend(train):
    """
    Line plot showing average daily sales by month across all years,
    revealing seasonal patterns.
    """
    train['month'] = train['date'].dt.month
    train['year'] = train['date'].dt.year
    
    # Monthly average by year
    monthly = train.groupby(['year', 'month'])['sales'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    cmap = plt.cm.viridis
    years = sorted(monthly['year'].unique())
    for i, year in enumerate(years):
        year_data = monthly[monthly['year'] == year]
        color = cmap(i / len(years))
        ax.plot(year_data['month'], year_data['sales'],
                marker='o', linewidth=2, markersize=6,
                label=str(year), color=color, alpha=0.85)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Daily Sales')
    ax.set_title('Monthly Sales Trend by Year (Seasonality Analysis)', fontweight='bold', pad=15)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(title='Year', loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'monthly_sales_trend.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: Store-Item Sales Heatmap
# ---------------------------------------------------------------------------
def plot_store_item_heatmap(train):
    """
    Heatmap showing average daily sales for each store-item combination.
    """
    pivot = train.groupby(['store', 'item'])['sales'].mean().reset_index()
    pivot_table = pivot.pivot(index='store', columns='item', values='sales')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    im = ax.imshow(pivot_table.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xlabel('Item ID')
    ax.set_ylabel('Store ID')
    ax.set_title('Average Daily Sales: Store × Item Heatmap', fontweight='bold', pad=15)
    
    ax.set_xticks(range(0, 50, 5))
    ax.set_xticklabels(range(1, 51, 5))
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(1, 11))
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Avg Daily Sales', fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'store_item_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 5: Weekly Sales Pattern
# ---------------------------------------------------------------------------
def plot_weekly_pattern(train):
    """
    Bar chart showing average sales by day of the week.
    """
    train['dayofweek'] = train['date'].dt.dayofweek
    
    weekly = train.groupby('dayofweek')['sales'].mean()
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    colors = [COLORS['primary']] * 5 + [COLORS['warm']] * 2  # Highlight weekends
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(day_labels, weekly.values, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Daily Sales')
    ax.set_title('Average Sales by Day of Week', fontweight='bold', pad=15)
    
    # Add value labels
    for bar, val in zip(bars, weekly.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], label='Weekday'),
        Patch(facecolor=COLORS['warm'], label='Weekend'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'weekly_sales_pattern.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 6: Time Series Sample
# ---------------------------------------------------------------------------
def plot_sample_timeseries(train):
    """
    Plot actual sales time series for a sample of store-item combinations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    samples = [(1, 1), (3, 15), (5, 25), (8, 45)]
    
    for ax, (store, item) in zip(axes.flatten(), samples):
        subset = train[(train['store'] == store) & (train['item'] == item)].sort_values('date')
        
        # Plot with rolling average overlay
        ax.plot(subset['date'], subset['sales'], alpha=0.3, color=COLORS['primary'], linewidth=0.5)
        ax.plot(subset['date'], subset['sales'].rolling(30).mean(),
                color=COLORS['warm'], linewidth=1.5, label='30-day MA')
        
        ax.set_title(f'Store {store}, Item {item}', fontweight='bold')
        ax.set_xlabel('')
        ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('Sample Time Series with 30-Day Moving Average', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'sample_timeseries.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Store Item Demand Forecasting — EDA & Visualizations")
    print("=" * 60)
    
    ensure_dirs()
    train, submission = load_data()
    
    print(f"\nTrain shape: {train.shape}")
    print(f"Date range: {train['date'].min()} to {train['date'].max()}")
    print(f"Stores: {train['store'].nunique()}, Items: {train['item'].nunique()}")
    print(f"Total store-item combinations: {train.groupby(['store','item']).ngroups}")
    
    print("\n--- Descriptive Statistics ---")
    print(train['sales'].describe().to_string())
    
    if submission is not None:
        print(f"\nSubmission shape: {submission.shape}")
        print(submission['sales'].describe().to_string())
    
    print("\nGenerating plots...")
    plot_feature_importance()
    plot_monthly_sales_trend(train)
    plot_store_item_heatmap(train)
    plot_weekly_pattern(train)
    plot_sample_timeseries(train)
    
    if submission is not None:
        plot_prediction_distribution(submission)
    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
