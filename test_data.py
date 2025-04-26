import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from prepreprocessing.data_loader import load_and_preprocess_data

def test_data_loading():
    """Test if data loading works correctly."""
    try:
        print("Testing data loader...")
        df = load_and_preprocess_data()
        
        print("\nSummary Statistics:")
        print(df.describe())
        
        # Save a quick plot of the data
        plt.figure(figsize=(12, 6))
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol]
            plt.plot(symbol_data['Date'], symbol_data['Close'], label=symbol)
        
        plt.title('Cryptocurrency Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price (Normalized)')
        plt.legend()
        plt.grid(True)
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/prices.png')
        print("\nSaved plot to plots/prices.png")
        
        print("\nDATA LOADING TEST: SUCCESS ✅")
        return True
    
    except Exception as e:
        print(f"\nDATA LOADING TEST: FAILED ❌")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_data_loading() 