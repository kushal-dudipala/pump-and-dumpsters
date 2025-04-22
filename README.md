# Pump and Dumpsters CS4644/CS7643

## Description

Pump and Dumpsters is a deep learning project for detecting anomalies (potential pump and dump schemes) in cryptocurrency price data. We implement and compare multiple deep learning approaches including LSTM, Autoencoder, and a hybrid CNN-LSTM model.

## Authors

Ronak Argawal, Kushal Dudipala, Rashmith Repala

## Project Status

✅ Complete:

- LSTM Model implementation
- Autoencoder implementation
- Hybrid CNN-LSTM Model implementation
- Data preparation utilities
- Evaluation framework
- Basic visualization
- Comprehensive visualization analysis
- Result comparison and insights

🔄 Underway:

- Running models on larger dataset
- Final report preparation

## Instructions

### Setup

1. Clone the repository
2. Set up a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) For macOS with Apple Silicon:

```bash
pip install tensorflow-macos tensorflow-metal
```

### Running the Code

```bash
# Run all models and evaluation
python sandbox.py

# Generate visualizations
python visualize_results.py
```

### Data

- Sample data is included in data/raw/sample_bitcoin.csv
- To download the full dataset (requires Kaggle account):

```bash
python data/kaggledownload.py
```

## Project Structure

- `models/`: Contains all model implementations
- `pumpdumpsters/`: Evaluation utilities and metrics
- `data/`: Data loading and preprocessing
- `notebooks/`: Analysis notebooks
  - `visualization_analysis.ipynb`: In-depth analysis of model results
- `plots/`: Output visualizations and comparison data

## See Also

See PROJECT_SUMMARY.md for a detailed overview of the project.

## To-Do List

- [x] **Kaggle** – Work on dataset _(Rashmih)_
- [x] **anomoloy_scan.py** – Find large jumps in data before feeding to models _(Kushal, Rashmith)_
- [x] **models/** – Write all model classes
  - [x] `lstm_model.py` – Implement LSTM model _(Kushal)_
  - [x] `auto_encoder.py` – Implement Auto Encoder model _(Kushal)_
- [x] **scripts/** – Write pace scrum script _(Kushal)_
- [x] **pumpdumpsters/** – Verify our evaluation scripts _Ronak_
- [ ] **write checkin** – Write our checkin proposal _Together_

## Future To-Do

- [ ] `hybrid_cnn_lstm.py` – Implement Hybrid CNN model _(Together)_
- [ ] **data_cleaning.py** – Clean data if necessary _(Ronak)_
- [ ] **feature_learning.py** – Implement feature learning if necessary _(Together)_

## Notes 3/26

- Need to fix visualizations in pumpdumpsters
- Added pace support, but we may end up using elsewhere.
