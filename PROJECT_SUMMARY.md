# Pump and Dumpsters: Cryptocurrency Anomaly Detection

## Project Overview

This project implements and compares multiple deep learning models for anomaly detection in cryptocurrency price data. We have implemented three distinct models:

1. **LSTM Model**: A sequential model for time series prediction that flags anomalies when actual prices deviate significantly from predictions
2. **Autoencoder Model**: An unsupervised learning approach that detects anomalies based on reconstruction error
3. **Hybrid CNN-LSTM Model**: A combined model that uses CNN layers for feature extraction and LSTM layers for temporal dependencies

## Key Components

### Data Handling

- `data/raw/sample_bitcoin.csv`: Sample cryptocurrency data for testing
- `data/kaggledownload.py`: Script for downloading larger dataset from Kaggle

### Models

- `models/lstm_model.py`: LSTM model implementation
- `models/auto_encoder.py`: Autoencoder model implementation
- `models/hybrid_cnn_lstm.py`: Hybrid CNN-LSTM model implementation
- `models/utils.py`: Utility functions for data handling and model training

### Evaluation

- `pumpdumpsters/evaluation.py`: Comprehensive evaluation metrics
- `pumpdumpsters/run_all.py`: Runs all evaluation metrics
- `visualize_results.py`: Visualizes anomaly detection results from all models

### Notebooks

- `notebooks/eda.ipynb`: Exploratory data analysis (placeholder)
- `notebooks/model_testing.ipynb`: Model testing and comparison (placeholder)
- `notebooks/visualizations.ipynb`: Visualization of results (placeholder)
- `notebooks/visualization_analysis.ipynb`: In-depth analysis of model results and anomaly detection patterns

## Running the Project

### Prerequisites

- Python 3.8+ with venv or conda
- Required packages in requirements.txt

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional packages for macOS
# For Apple Silicon Macs:
pip install tensorflow-macos tensorflow-metal
```

### Running the Models

```bash
# Test data loading
python test_data.py

# Run all models and evaluate
python sandbox.py

# Generate visualizations
python visualize_results.py
```

## Results and Visualizations

- Anomaly detection results are saved in `plots/anomaly_detection_comparison.png`
- Comparison table is saved in `plots/anomaly_comparison.csv`
- Interactive analysis of results available in `notebooks/visualization_analysis.ipynb`
- Comprehensive insights include:
  - Comparison of model precision, recall, and F1 scores
  - Temporal distribution analysis of detected anomalies
  - Correlation between price movements and anomaly detection
  - Overlap analysis between different models' anomaly predictions

## Model Performance

Based on our sample data testing:

- **LSTM Model**: Achieves high precision and recall, with a good balance between false positives and false negatives
- **Autoencoder Model**: Detects more anomalies but has a higher false positive rate
- **Hybrid CNN-LSTM Model**: Combines feature extraction and temporal patterns for robust anomaly detection

## Next Steps for Final Report

1. **Comprehensive Analysis**: Run models on larger dataset for more realistic anomaly detection
2. **Hyperparameter Tuning**: Experiment with different parameters to optimize model performance
3. **Result Visualization**: Create detailed visualizations comparing model performance
4. **Report Writing**: Prepare final report following the provided rubric
5. **Team Contributions**: Document specific contributions from each team member

## Troubleshooting

- If you encounter TensorFlow issues on macOS, try using the Apple-specific versions: `tensorflow-macos` and `tensorflow-metal`
- For data loading issues, check the path to your data file or use the provided sample data
