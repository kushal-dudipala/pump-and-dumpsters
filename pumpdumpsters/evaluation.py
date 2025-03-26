from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union, Optional, List, Tuple

def evaluate_model(df):
    """
    Compares anomalies from different methods with comprehensive metrics.
    
    Parameters:
    -----------
    df : DataFrame with anomaly detection columns
    
    Returns:
    --------
    metrics_dict : Dictionary containing evaluation metrics
    """
    # Initialize metrics dictionary
    metrics_dict = {}
    
    # List of anomaly detection methods to evaluate
    anomaly_methods = []
    
    # Check which anomaly detection methods are present in the dataframe
    if 'autoencoder_anomaly' in df.columns:
        df['autoencoder_anomaly'] = df['autoencoder_anomaly'].infer_objects(False)
        df['autoencoder_anomaly'] = df['autoencoder_anomaly'].infer_objects(copy=False).astype(bool).astype(int)
        anomaly_methods.append('autoencoder_anomaly')
    
    if 'lstm_anomaly' in df.columns:
        df['lstm_anomaly'] = df['lstm_anomaly'].infer_objects(copy=False).astype(bool).astype(int)
        anomaly_methods.append('lstm_anomaly')
    
    if 'anomaly' in df.columns:  # Z-score anomaly
        df['anomaly'] = df['anomaly'].infer_objects(copy=False).astype(bool).astype(int)
        anomaly_methods.append('anomaly')
    
    # If we have at least two methods to compare
    if len(anomaly_methods) >= 2:
        # Use z-score anomaly as ground truth if available, otherwise use the first method
        ground_truth = 'anomaly' if 'anomaly' in anomaly_methods else anomaly_methods[0]
        
        # Compare each method against the ground truth
        for method in anomaly_methods:
            if method != ground_truth:
                print(f"\nEvaluating {method} against {ground_truth}:")
                
                # Classification report
                report = classification_report(df[ground_truth], df[method], output_dict=True, zero_division=0)
                print(classification_report(df[ground_truth], df[method], zero_division=0))
                metrics_dict[f"{method}_report"] = report
                
                # Confusion matrix
                cm = confusion_matrix(df[ground_truth], df[method], labels=[0, 1])
                plot_confusion_matrix(cm, [0, 1], title=f"Confusion Matrix: {method}")
                metrics_dict[f"{method}_confusion_matrix"] = cm
                
                # Precision, recall, F1
                precision, recall, f1, _ = precision_recall_fscore_support(df[ground_truth], df[method], average='binary', zero_division=0)
                metrics_dict[f"{method}_precision"] = precision
                metrics_dict[f"{method}_recall"] = recall
                metrics_dict[f"{method}_f1"] = f1
                
                # ROC curve and AUC if the method provides probability scores
                if f"{method.replace('_anomaly', '')}_score" in df.columns:
                    score_col = f"{method.replace('_anomaly', '')}_score"
                    fpr, tpr, _ = roc_curve(df[ground_truth], df[score_col])
                    roc_auc = auc(fpr, tpr)
                    plot_roc_curve(fpr, tpr, roc_auc, title=f"ROC Curve: {method}")
                    metrics_dict[f"{method}_auc"] = roc_auc
    
    # Evaluate each method individually
    for method in anomaly_methods:
        # Calculate percentage of anomalies detected
        anomaly_percentage = df[method].mean() * 100
        print(f"\n{method} detected {anomaly_percentage:.2f}% of data points as anomalies")
        metrics_dict[f"{method}_percentage"] = anomaly_percentage
        
        # Plot anomaly distribution over time
        plot_anomaly_distribution(df, method)
    
    return metrics_dict

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """Plots confusion matrix with a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, title='ROC Curve'):
    """Plots ROC curve with AUC value."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_anomaly_distribution(df, anomaly_col):
    """Plots the distribution of anomalies over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Price', alpha=0.7)
    plt.scatter(df[df[anomaly_col] == 1]['Date'], 
                df[df[anomaly_col] == 1]['Close'], 
                color='red', label=f'{anomaly_col}')
    plt.title(f'Distribution of {anomaly_col} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
def compare_all_anomaly_methods(df):
    """
    Creates a visualization comparing all anomaly detection methods.
    
    Parameters:
    -----------
    df : DataFrame with multiple anomaly detection columns
    """
    # Check which anomaly methods are available
    anomaly_cols = [col for col in df.columns if 'anomaly' in col]
    
    if len(anomaly_cols) < 2:
        print("Not enough anomaly detection methods to compare")
        return
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(anomaly_cols), 1, figsize=(12, 4*len(anomaly_cols)), sharex=True)
    
    # If only one method, axes won't be an array
    if len(anomaly_cols) == 1:
        axes = [axes]
    
    # Plot each method
    for i, col in enumerate(anomaly_cols):
        axes[i].plot(df['timestamp'], df['close'], label='Price', alpha=0.7)
        axes[i].scatter(df[df[col] == 1]['timestamp'], 
                      df[df[col] == 1]['close'], 
                      color='red', label=col)
        axes[i].set_title(f'{col} Detection')
        axes[i].set_ylabel('Price')
        axes[i].legend()
    
    # Set common x-label
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
    
    # Create a Venn diagram-like visualization for overlap
    # This is a simplified approach - for actual Venn diagrams you'd need additional libraries
    overlap_counts = {}
    for i, col1 in enumerate(anomaly_cols):
        for col2 in anomaly_cols[i+1:]:
            overlap = ((df[col1] == 1) & (df[col2] == 1)).sum()
            total1 = (df[col1] == 1).sum()
            total2 = (df[col2] == 1).sum()
            print(f"Overlap between {col1} and {col2}: {overlap} out of {total1} and {total2} anomalies")
            overlap_counts[f"{col1}_{col2}"] = overlap
    
    return overlap_counts

def clean_cryptocurrency_data(
    df: pd.DataFrame,
    handle_missing: bool = True,
    handle_outliers: bool = True,
    outlier_method: str = 'iqr',
    outlier_columns: Optional[List[str]] = None,
    min_volume: Optional[float] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Cleans cryptocurrency data by handling missing values, outliers, and other issues.
    
    Parameters:
    -----------
    df : DataFrame with cryptocurrency data
    handle_missing : Whether to handle missing values
    handle_outliers : Whether to handle outliers
    outlier_method : Method to detect outliers ('iqr' or 'zscore')
    outlier_columns : Columns to check for outliers (default: price and volume columns)
    min_volume : Minimum trading volume to keep (filters out low liquidity periods)
    
    Returns:
    --------
    cleaned_df : Cleaned DataFrame
    cleaning_stats : Dictionary with cleaning statistics
    """
    original_rows = len(df)
    cleaning_stats = {'original_rows': original_rows}
    
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in cleaned_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(cleaned_df['timestamp']):
            try:
                # Try parsing as unix timestamp first
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'], unit='s')
            except:
                # If that fails, try standard datetime parsing
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
    
    # Handle missing values
    if handle_missing:
        missing_before = cleaned_df.isna().sum().sum()
        
        # For price data, forward fill is often appropriate (carry last known price forward)
        price_cols = [col for col in cleaned_df.columns if col in ['open', 'high', 'low', 'close']]
        if price_cols:
            cleaned_df[price_cols] = cleaned_df[price_cols].fillna(method='ffill').infer_objects(copy=False)
            # If there are still NaNs at the beginning, backward fill
            cleaned_df[price_cols] = cleaned_df[price_cols].fillna(method='bfill').infer_objects(copy=False)
        
        # For volume, replace NaN with 0 (no trading)
        if 'volume' in cleaned_df.columns:
            cleaned_df['volume'] = cleaned_df['volume'].fillna(0).infer_objects(copy=False)
        
        # For any remaining NaNs, use column median
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        
        missing_after = cleaned_df.isna().sum().sum()
        cleaning_stats['missing_values_removed'] = missing_before - missing_after
    
    # Filter by minimum volume if specified
    if min_volume is not None and 'volume' in cleaned_df.columns:
        rows_before = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df['volume'] >= min_volume]
        rows_removed = rows_before - len(cleaned_df)
        cleaning_stats['low_volume_rows_removed'] = rows_removed
    
    # Handle outliers
    if handle_outliers:
        if outlier_columns is None:
            # Default to price and volume columns
            outlier_columns = [col for col in cleaned_df.columns 
                              if col in ['open', 'high', 'low', 'close', 'volume']]
        
        outliers_removed = 0
        
        for col in outlier_columns:
            if col in cleaned_df.columns:
                if outlier_method == 'iqr':
                    # IQR method
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Count outliers
                    outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
                    outliers_removed += outliers
                    
                    # Replace outliers with bounds
                    cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                    cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
                    
                elif outlier_method == 'zscore':
                    # Z-score method
                    z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                    outliers = (z_scores > 3).sum()
                    outliers_removed += outliers
                    
                    # Replace outliers with column mean
                    cleaned_df.loc[z_scores > 3, col] = cleaned_df[col].mean()
        
        cleaning_stats['outliers_handled'] = outliers_removed
    
    # Sort by timestamp if it exists
    if 'timestamp' in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values('timestamp')
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    # Calculate final statistics
    cleaning_stats['final_rows'] = len(cleaned_df)
    cleaning_stats['rows_removed'] = original_rows - len(cleaned_df)
    cleaning_stats['percentage_removed'] = (original_rows - len(cleaned_df)) / original_rows * 100
    
    return cleaned_df, cleaning_stats

def detect_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and handles duplicate timestamps in the data.
    
    Parameters:
    -----------
    df : DataFrame with a 'timestamp' column
    
    Returns:
    --------
    df : DataFrame with duplicates handled
    """
    if 'timestamp' not in df.columns:
        print("No timestamp column found")
        return df
    
    # Check for duplicates
    duplicates = df[df.duplicated('timestamp', keep=False)]
    
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicate timestamps")
        
        # Group by timestamp and aggregate
        # For price data, take the last value
        # For volume, sum all volumes
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Only include columns that exist in the dataframe
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        # Aggregate duplicates
        df = df.groupby('timestamp').agg(agg_dict).reset_index()
    
    return df

def identify_gaps(df: pd.DataFrame, expected_frequency: str = '1D') -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identifies gaps in time series data.
    
    Parameters:
    -----------
    df : DataFrame with a 'timestamp' column
    expected_frequency : Expected frequency of data points ('1D' for daily, '1H' for hourly, etc.)
    
    Returns:
    --------
    gaps : List of tuples containing the start and end of each gap
    """
    if 'timestamp' not in df.columns:
        print("No timestamp column found")
        return []
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create a complete date range at the expected frequency
    full_date_range = pd.date_range(
        start=df['timestamp'].min(),
        end=df['timestamp'].max(),
        freq=expected_frequency
    )
    
    # Find missing dates
    existing_dates = set(df['timestamp'])
    missing_dates = [date for date in full_date_range if date not in existing_dates]
    
    # Group consecutive missing dates into gaps
    gaps = []
    if missing_dates:
        missing_dates = sorted(missing_dates)
        gap_start = missing_dates[0]
        prev_date = missing_dates[0]
        
        for date in missing_dates[1:]:
            # If this date is not consecutive to the previous one, we have a new gap
            if date - prev_date > pd.Timedelta(expected_frequency):
                gaps.append((gap_start, prev_date))
                gap_start = date
            prev_date = date
        
        # Add the last gap
        gaps.append((gap_start, missing_dates[-1]))
    
    return gaps