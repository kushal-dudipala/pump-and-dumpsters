import os

from pumpdumpsters.eda import plot_price_trends, plot_volume_trends
from pumpdumpsters.anomaly_detection import detect_anomalies_zscore, plot_zscore_anomalies
from pumpdumpsters.mean_reversion import apply_mean_reversion_strategy, plot_mean_reversion
from pumpdumpsters.evaluation import evaluate_model

def run_all_evaluation_metrics(df, project_root):
    """
    Main function to run the entire pipeline for cryptocurrency analysis.
    This includes loading data, performing EDA, training models, and evaluating performance.
    """

    # Plots folder
    plots_folder = os.path.join(project_root, "plots")
    
    # Perform Exploratory Data Analysis
    print("Performing EDA...")
    plot_price_trends(df, plots_folder)
    plot_volume_trends(df, plots_folder)

    # Baseline Anomaly Detection
    print("Detecting anomalies with Z-score...")
    df = detect_anomalies_zscore(df)
    plot_zscore_anomalies(df, plots_folder)

    # Apply Mean Reversion Strategy
    print("Applying mean reversion strategy...")
    df = apply_mean_reversion_strategy(df)
    plot_mean_reversion(df, plots_folder)

    # Evaluate Model Performance
    print("Evaluating Model Performance...")
    evaluate_model(df, plots_folder)

