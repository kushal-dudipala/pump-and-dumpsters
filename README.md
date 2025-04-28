# Pump and Dumpsters CS4644/CS7643

## Description

Pump and Dumpsters is a deep learning project for detecting anomalies (potential pump and dump schemes) in cryptocurrency price data. We implement and compare multiple deep learning approaches including LSTM, Autoencoder, and a hybrid CNN-LSTM model.

## Authors

Ronak Argawal, Kushal Dudipala, Rashmith Repala

### Setup

1. Clone the repository
2. Set up a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
which pip # Ensure proper pip installation
pip install -r requirements.txt
```
### Running the Code

```bash
# Run a test example of models and evaluation
python sandbox.py

# Generate visualizations
python visualize_results.py
```

### Data

To download the full dataset (requires Kaggle account):

1. Create a `kaggle.json` file with your Kaggle API key. You can obtain the API key from your Kaggle account settings.

2. Place the `kaggle.json` file in the `~/.kaggle/` directory (or the equivalent directory on your system).

3. Run the following command to download the dataset:

```bash
python data/kaggledownload.py
```

4. Combine the dataset by running: 
```bash
python data/kaggledownload.py
```

## Project Structure

- `models/`: Contains all model implementations
  - `saved_hyperparams/`: Stores JSON's of optimal hyperparameters
- `pumpdumpsters/`: Evaluation utilities and metrics
- `data/`: Data loading and preprocessing
- `notebooks/`: Analysis notebooks
  - `visualization_analysis.ipynb`: In-depth analysis of model results
- `plots/`: Output visualizations and comparison data
- `scripts/`: Contains training and sweep scripts, as well as PACE



## How to Use Code

### Hyperparameter Sweep

To perform a hyperparameter sweep for any of the models (`lstm`, `cnn`, or `hybrid`), use the following command:

```bash
python scripts/hyperparam_sweep.py --model <model_name>
```

Replace `<model_name>` with one of the following:
- `lstm`: For the LSTM model
- `cnn`: For the Autoencoder model
- `hybrid`: For the Hybrid CNN-LSTM model

The script will run a grid search over predefined hyperparameters and save the best hyperparameters and their corresponding F1 score in a JSON file.

### Training Models Individually

To train a specific model with custom hyperparameters, use the `train_models.py` script:

```bash
python scripts/train_models.py --model <model_name> [options]
```

Replace `<model_name>` with one of the following:
- `lstm`: For the LSTM model
- `cnn`: For the Autoencoder model
- `hybrid`: For the Hybrid CNN-LSTM model

Options:
- `--seq_len`: Sequence length for LSTM/Hybrid models (default: 30)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate for the optimizer (default: 0.001)
- `--dropout_rate`: Dropout rate to prevent overfitting (default: 0.2)
- `--cnn_filters`: Number of filters in the CNN layer (for Hybrid model, default: 64)
- `--cnn_kernel_size`: Kernel size for the CNN layer (for Hybrid model, default: 3)
- `--lstm_units`: Number of units in the LSTM layer (for LSTM/Hybrid models, default: 64)
- `--test_size`: Fraction of data used for testing (default: 0.2)
- `--filename`: Path to a custom CSV file for training (default: merged dataset in `data/processed/`)

Example:
```bash
python scripts/train_models.py --model lstm --seq_len 50 --epochs 30 --learning_rate 0.001
```

### Running All Models Locally

To train all models (`lstm`, `cnn`, and `hybrid`) using their optimal hyperparameters and evaluate them, use the `run_all_models.py` script:

```bash
python scripts/run_all_models.py
```

This script will:
1. Load the merged cryptocurrency dataset from `data/processed/merged_cryptos.csv`.
2. Train each model using predefined optimal hyperparameters.
3. Detect anomalies using each model.
4. Evaluate the results using the evaluation framework.
5. Print the evaluation metrics for all models.

```bash
# Example output:
=== TRAINING LSTM MODEL ===
LSTM anomaly detection complete.

=== TRAINING CNN AUTOENCODER MODEL ===
Autoencoder anomaly detection complete.

=== TRAINING HYBRID CNN-LSTM MODEL ===
Hybrid CNN-LSTM anomaly detection complete.

=== RUNNING EVALUATION METRICS ===
Evaluation Results:
{...}
```

### Running on PACE 

To run training or sweeps on Georgia Tech's PACE cluster:

```bash
# Step 1: Submit a job (example)
sbatch scripts/run_pace.sbatch

# Step 2: Monitor your job
squeue -u YOUR_USERNAME

# Step 3: Check job output
tail -f sweep<JOBID>.out
```


**Notes:**
- Modify `scripts/run_pace.sbatch` to decide which script to run. hyper_param_sweep_pace.sh runs a full hyperparameter sweep and saves the optimal hyperperameters in JSON's.


### Data Analysis
* The mean reversion, EDA, and anomaly detection visualizations are controlled by the `pumpdumpsters/run_all.py` file. 
* All evaluations, confusion matrix, ROC, and anomaly distribution visualizations are controlled by the in`pumpdumpsters/evaluation.py`.
* **ALL VISUALIZATIONS** used in this paper stored in the `plots/` folder. 


