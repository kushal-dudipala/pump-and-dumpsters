import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from prepreprocessing.data_loader import load_and_preprocess_data

# Run the whole pipeline first
from sandbox import run_model_pipeline

def visualize_anomalies(df):
    """
    Visualize the anomalies detected by different models.
    
    Parameters:
    -----------
    df : DataFrame with anomaly flags from various models
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot settings
    plt.figure(figsize=(14, 10))
    
    # Get unique symbols
    symbols = df['Symbol'].unique()
    
    # Create subplots for each symbol
    for i, symbol in enumerate(symbols):
        symbol_data = df[df['Symbol'] == symbol].copy()
        
        # Get anomaly columns
        anomaly_cols = [col for col in symbol_data.columns if 'anomaly' in col.lower()]
        
        # Create subplot
        plt.subplot(len(symbols), 1, i+1)
        
        # Plot price
        plt.plot(symbol_data['Date'], symbol_data['Close'], label=f'{symbol} Price', color='black', alpha=0.7)
        
        # Plot anomalies for each method with different markers and colors
        colors = ['red', 'blue', 'green', 'purple']
        markers = ['o', 's', '^', 'x']
        
        for j, col in enumerate(anomaly_cols):
            anomaly_points = symbol_data[symbol_data[col] == True]
            if len(anomaly_points) > 0:
                plt.scatter(anomaly_points['Date'], 
                          anomaly_points['Close'], 
                          color=colors[j % len(colors)], 
                          marker=markers[j % len(markers)],
                          s=100, 
                          label=f'{col}',
                          alpha=0.7)
        
        plt.title(f'Anomaly Detection for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Close Price (Normalized)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/anomaly_detection_comparison.png')
    print(f"Saved anomaly visualization to plots/anomaly_detection_comparison.png")
    
    # Create a comparison table of anomalies
    comparison_table = pd.DataFrame()
    comparison_table['Date'] = df['Date']
    comparison_table['Symbol'] = df['Symbol']
    comparison_table['Close'] = df['Close']
    
    for col in [c for c in df.columns if 'anomaly' in c.lower()]:
        comparison_table[col] = df[col]
    
    # Save to CSV
    comparison_table.to_csv('plots/anomaly_comparison.csv', index=False)
    print(f"Saved anomaly comparison table to plots/anomaly_comparison.csv")
    
    return comparison_table

if __name__ == "__main__":
    # Run the model pipeline
    success = run_model_pipeline()
    
    if success:
        # Load data with anomaly flags
        df = load_and_preprocess_data()
        
        # Add some synthetic anomalies for demonstration
        if 'anomaly' not in df.columns:
            # Create a simple anomaly column based on price jumps
            df['anomaly'] = False
            df.loc[df['Close'].diff().abs() > 0.02, 'anomaly'] = True
        
        # Visualize results
        comparison = visualize_anomalies(df)
        print("\nAnomaly Counts by Method:")
        for col in [c for c in comparison.columns if 'anomaly' in c.lower()]:
            anomaly_count = comparison[col].sum()
            total_count = len(comparison)
            print(f"{col}: {anomaly_count} anomalies ({anomaly_count/total_count:.1%} of data)")
    else:
        print("Model pipeline failed. Please check for errors.") 