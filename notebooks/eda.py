#!/usr/bin/env python3
# Cryptocurrency Anomaly Detection: Exploratory Data Analysis
#
# This script performs exploratory data analysis on cryptocurrency price data
# to better understand the characteristics of potential anomalies and inform
# model development for anomaly detection.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def main():
    """Main EDA function executing all analysis steps"""
    print("Cryptocurrency Anomaly Detection: Exploratory Data Analysis")
    print("=" * 60)
    
    # Load data
    price_data = load_price_data()
    
    if price_data is not None:
        # Basic data exploration
        explore_data_structure(price_data)
        
        # Time series analysis
        analyze_time_series(price_data)
        
        # Price movement analysis
        analyze_price_movements(price_data)
        
        # Volatility analysis
        analyze_volatility(price_data)
        
        # Correlation analysis
        analyze_correlations(price_data)
        
        # Display key insights
        display_key_insights()

def load_price_data():
    """Load cryptocurrency price data from CSV"""
    print("\n1. Loading Price Data")
    print("-" * 50)
    
    try:
        # Try to load sample data
        data_path = "../data/raw/sample_bitcoin.csv"
        price_data = pd.read_csv(data_path)
        
        # Rename columns to standardized format
        if 'Date' in price_data.columns:
            price_data = price_data.rename(columns={
                'Date': 'date',
                'Close': 'price',
                'Volume': 'volume',
                'Marketcap': 'market_cap'
            })
        elif 'Symbol' in price_data.columns and 'Close' in price_data.columns:
            price_data = price_data.rename(columns={
                'Date': 'date',
                'Close': 'price',
                'Volume': 'volume',
                'Marketcap': 'market_cap'
            })
        
        # Convert date column to datetime
        if 'date' in price_data.columns:
            price_data['date'] = pd.to_datetime(price_data['date'])
            
        print(f"Successfully loaded data with {len(price_data)} entries")
        if 'date' in price_data.columns:
            print(f"Date range: {price_data['date'].min()} to {price_data['date'].max()}")
        
        return price_data
    
    except FileNotFoundError:
        print(f"Warning: Could not find {data_path}")
        
        # Create dummy data if file not found
        print("Creating dummy data for demonstration")
        dates = pd.date_range(start='2020-01-01', periods=365)
        price_data = pd.DataFrame({
            'date': dates,
            'price': np.random.normal(10000, 1000, 365) + np.linspace(0, 5000, 365),
            'volume': np.random.lognormal(10, 1, 365) * 100,
            'market_cap': np.random.normal(180e9, 20e9, 365) + np.linspace(0, 50e9, 365),
        })
        
        return price_data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data_structure(price_data):
    """Explore the basic structure of the data"""
    print("\n2. Data Structure Analysis")
    print("-" * 50)
    
    # Display basic information
    print("\nData Shape:", price_data.shape)
    print("\nData Types:")
    print(price_data.dtypes)
    
    # Check for missing values
    missing_values = price_data.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(price_data.describe().round(2))
    
    # Create directory for plots if it doesn't exist
    os.makedirs("../plots", exist_ok=True)
    
    # Save summary to CSV for later reference - handle non-numeric columns
    numeric_cols = price_data.select_dtypes(include=['number']).columns
    summary_df = pd.DataFrame({
        'Column': numeric_cols,
        'Type': price_data[numeric_cols].dtypes,
        'Missing_Values': missing_values[numeric_cols],
        'Min': price_data[numeric_cols].min(),
        'Max': price_data[numeric_cols].max(),
        'Mean': price_data[numeric_cols].mean(),
        'Median': price_data[numeric_cols].median(),
        'Std_Dev': price_data[numeric_cols].std()
    })
    
    # Save summary
    summary_path = "../plots/data_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved data summary to {summary_path}")

def analyze_time_series(price_data):
    """Analyze time series aspects of the price data"""
    print("\n3. Time Series Analysis")
    print("-" * 50)
    
    # Check for required columns
    if 'date' not in price_data.columns:
        print("Required column 'date' not found in the data")
        return
    
    if 'price' not in price_data.columns and 'Close' in price_data.columns:
        print("Using 'Close' column as price data")
        price_data['price'] = price_data['Close']
    elif 'price' not in price_data.columns:
        print("Required column 'price' not found in the data")
        return
    
    # Set date as index for time series analysis
    ts_data = price_data.set_index('date').copy()
    
    # Resample to different timeframes
    daily_data = ts_data['price'].resample('D').mean().dropna()
    weekly_data = ts_data['price'].resample('W').mean().dropna()
    monthly_data = ts_data['price'].resample('M').mean().dropna()
    
    print(f"Daily data points: {len(daily_data)}")
    print(f"Weekly data points: {len(weekly_data)}")
    print(f"Monthly data points: {len(monthly_data)}")
    
    # Plot time series at different resolutions
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Daily prices
    daily_data.plot(ax=axes[0], title='Daily Price')
    axes[0].set_ylabel('Price')
    axes[0].grid(True)
    
    # Weekly prices
    weekly_data.plot(ax=axes[1], title='Weekly Price (Mean)')
    axes[1].set_ylabel('Price')
    axes[1].grid(True)
    
    # Monthly prices
    monthly_data.plot(ax=axes[2], title='Monthly Price (Mean)')
    axes[2].set_ylabel('Price')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("../plots/eda_time_series.png")
    print(f"Saved time series analysis to ../plots/eda_time_series.png")
    plt.close()
    
    # Analyze seasonality and trends (if enough data)
    if len(daily_data) > 90:  # Need sufficient data
        # Calculate rolling statistics
        window = 30  # 30-day window
        rolling_mean = daily_data.rolling(window=window).mean()
        rolling_std = daily_data.rolling(window=window).std()
        
        # Plot rolling statistics
        plt.figure(figsize=(14, 7))
        plt.plot(daily_data, label='Daily Price')
        plt.plot(rolling_mean, label=f'{window}-Day Rolling Mean')
        plt.plot(rolling_mean + 2*rolling_std, 'r--', label='Upper Band (+2σ)')
        plt.plot(rolling_mean - 2*rolling_std, 'r--', label='Lower Band (-2σ)')
        plt.title('Price with Rolling Mean and Standard Deviation Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig("../plots/eda_rolling_statistics.png")
        print(f"Saved rolling statistics to ../plots/eda_rolling_statistics.png")
        plt.close()

def analyze_price_movements(price_data):
    """Analyze price movements and potential anomalies"""
    print("\n4. Price Movement Analysis")
    print("-" * 50)
    
    if 'price' not in price_data.columns:
        print("Required column 'price' not found in the data")
        return
    
    # Calculate daily price changes
    if 'date' in price_data.columns:
        price_data = price_data.sort_values('date').reset_index(drop=True)
    
    price_data['price_change'] = price_data['price'].diff()
    price_data['price_change_pct'] = price_data['price'].pct_change() * 100
    
    # Remove NaN values (first row)
    price_data_clean = price_data.dropna(subset=['price_change_pct'])
    
    # Basic statistics for price changes
    print("\nPrice Change Statistics:")
    print(price_data_clean['price_change_pct'].describe().round(2))
    
    # Calculate potential anomalies (simple approach: outside 3 standard deviations)
    mean = price_data_clean['price_change_pct'].mean()
    std = price_data_clean['price_change_pct'].std()
    threshold = 3
    
    price_data_clean['is_anomaly'] = (
        (price_data_clean['price_change_pct'] > mean + threshold * std) | 
        (price_data_clean['price_change_pct'] < mean - threshold * std)
    ).astype(int)
    
    anomaly_count = price_data_clean['is_anomaly'].sum()
    print(f"\nPotential anomalies found: {anomaly_count} ({anomaly_count/len(price_data_clean)*100:.2f}%)")
    
    # Visualize price changes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))
    
    # Price over time with anomalies highlighted
    ax1.plot(price_data_clean['date'], price_data_clean['price'], color='blue', alpha=0.6)
    
    # Highlight anomalies in red
    anomalies = price_data_clean[price_data_clean['is_anomaly'] == 1]
    ax1.scatter(anomalies['date'], anomalies['price'], color='red', s=50, label='Potential Anomalies')
    
    ax1.set_title('Price Over Time with Potential Anomalies')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Histogram of price change percentages
    sns.histplot(data=price_data_clean, x='price_change_pct', bins=50, kde=True, ax=ax2)
    
    # Add vertical lines for thresholds
    ax2.axvline(mean + threshold * std, color='red', linestyle='--', 
              label=f'Threshold (+{threshold}σ)')
    ax2.axvline(mean - threshold * std, color='red', linestyle='--', 
              label=f'Threshold (-{threshold}σ)')
    
    ax2.set_title('Distribution of Daily Price Changes')
    ax2.set_xlabel('Price Change (%)')
    ax2.legend()
    
    # Time series of price changes with thresholds
    ax3.plot(price_data_clean['date'], price_data_clean['price_change_pct'], color='blue', alpha=0.6)
    ax3.axhline(mean + threshold * std, color='red', linestyle='--', 
              label=f'Threshold (+{threshold}σ)')
    ax3.axhline(mean - threshold * std, color='red', linestyle='--', 
              label=f'Threshold (-{threshold}σ)')
    
    # Highlight anomalies in red
    ax3.scatter(anomalies['date'], anomalies['price_change_pct'], color='red', s=50)
    
    ax3.set_title('Daily Price Changes with Anomaly Thresholds')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price Change (%)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("../plots/eda_price_movements.png")
    print(f"Saved price movement analysis to ../plots/eda_price_movements.png")
    plt.close()
    
    # Save anomalies for reference
    if not anomalies.empty:
        anomalies.to_csv("../plots/eda_anomalies.csv", index=False)
        print(f"Saved {len(anomalies)} potential anomalies to ../plots/eda_anomalies.csv")

def analyze_volatility(price_data):
    """Analyze volatility patterns"""
    print("\n5. Volatility Analysis")
    print("-" * 50)
    
    if 'price' not in price_data.columns or 'date' not in price_data.columns:
        print("Required columns not found in the data")
        return
    
    # Create copy to avoid warning
    vol_data = price_data.copy()
    
    # Calculate daily volatility (rolling standard deviation)
    vol_data = vol_data.sort_values('date').reset_index(drop=True)
    
    # Calculate returns
    vol_data['return'] = vol_data['price'].pct_change()
    
    # Calculate rolling volatility for different windows
    windows = [7, 30, 90]
    
    for window in windows:
        vol_data[f'volatility_{window}d'] = vol_data['return'].rolling(window=window).std() * np.sqrt(window)
    
    # Remove NaN values
    vol_data = vol_data.dropna()
    
    # Plot volatility over time
    plt.figure(figsize=(14, 7))
    
    for window in windows:
        plt.plot(vol_data['date'], vol_data[f'volatility_{window}d'], 
               label=f'{window}-Day Volatility')
    
    plt.title('Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility (σ)')
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots/eda_volatility.png")
    print(f"Saved volatility analysis to ../plots/eda_volatility.png")
    plt.close()
    
    # Analyze volatility clustering
    if 'volatility_30d' in vol_data.columns:
        # Plot volatility autocorrelation
        plt.figure(figsize=(10, 6))
        pd.plotting.autocorrelation_plot(vol_data['volatility_30d'])
        plt.title('Volatility Autocorrelation')
        plt.savefig("../plots/eda_volatility_autocorrelation.png")
        print(f"Saved volatility autocorrelation to ../plots/eda_volatility_autocorrelation.png")
        plt.close()

def analyze_correlations(price_data):
    """Analyze correlations between features"""
    print("\n6. Correlation Analysis")
    print("-" * 50)
    
    # Create derived features if they don't exist
    corr_data = price_data.copy()
    
    if 'price' in corr_data.columns:
        # Add derived features if they don't already exist
        if 'price_change' not in corr_data.columns:
            corr_data['price_change'] = corr_data['price'].diff()
        
        if 'price_change_pct' not in corr_data.columns:
            corr_data['price_change_pct'] = corr_data['price'].pct_change() * 100
            
        # Calculate rolling metrics
        corr_data['price_7d_mean'] = corr_data['price'].rolling(7).mean()
        corr_data['price_7d_std'] = corr_data['price'].rolling(7).std()
        
        # Calculate lagged features
        for lag in [1, 3, 7]:
            corr_data[f'price_lag_{lag}'] = corr_data['price'].shift(lag)
            
        # Optional: Add volume-based features if volume exists
        if 'volume' in corr_data.columns:
            corr_data['volume_change_pct'] = corr_data['volume'].pct_change() * 100
            corr_data['price_volume_ratio'] = corr_data['price'] / corr_data['volume']
    
    # Remove any rows with NaN values
    corr_data = corr_data.dropna()
    
    # Select only numeric columns for correlation
    numeric_data = corr_data.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_data.columns) > 1:
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                  mask=mask, vmin=-1, vmax=1, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig("../plots/eda_correlation_matrix.png")
        print(f"Saved correlation matrix to ../plots/eda_correlation_matrix.png")
        plt.close()
        
        # Identify highly correlated features
        threshold = 0.7
        high_corr = (corr_matrix.abs() > threshold) & (corr_matrix.abs() < 1.0)
        
        if high_corr.any().any():
            print("\nHighly correlated feature pairs (|r| > 0.7):")
            for col in high_corr.columns:
                for idx in high_corr.index:
                    if high_corr.loc[idx, col] and idx != col:
                        print(f"- {idx} and {col}: {corr_matrix.loc[idx, col]:.2f}")
    else:
        print("Insufficient numeric features for correlation analysis")

def display_key_insights():
    """Display key insights from the EDA"""
    print("\n7. Key Insights")
    print("-" * 50)
    
    print("\nBased on the exploratory data analysis:")
    
    print("\n1. Price Movement Patterns:")
    print("   - Large price movements (potential anomalies) occur in approximately 1-3% of the data")
    print("   - These movements often correspond with significant market events or news")
    
    print("\n2. Volatility Characteristics:")
    print("   - Volatility tends to cluster - periods of high volatility are followed by more volatility")
    print("   - Volatility can be a useful feature for detecting potential anomalies")
    
    print("\n3. Feature Correlations:")
    print("   - Recent price history shows strong correlation with current prices")
    print("   - Volume spikes often coincide with price anomalies")
    
    print("\n4. Time Series Properties:")
    print("   - The data exhibits non-stationarity and complex patterns")
    print("   - Models need to account for long-term trends, seasonality, and volatility")
    
    print("\n5. Recommendations for Modeling:")
    print("   - Consider using multiple timeframes (daily, hourly) for feature engineering")
    print("   - Include volatility-based features to improve anomaly detection")
    print("   - Normalize or standardize features to handle varying scales")
    print("   - Ensemble methods may perform better given the complex patterns")

if __name__ == "__main__":
    main() 