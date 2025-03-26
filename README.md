# Pump and Dumpsters CS4644/CS7643

## Description
Pump and Dumpsters 

## Authors
Ronak Argawal, Kushal Dudipala, Rashmith Repala

## Instructions

1) Open your terminal and navigate to the `pumpdumpsters` folder, then install the module in local editable mode (if editing files in `pumpdumpsters`, the module will reactievly update):

   ```bash
   cd pumpdumpsters
   pip install -e .
   ```

   If you are not running in editable mode, use the following instead:

   ```bash
   cd pumpdumpsters
   pip install .
   ```


## To-Do List

- [x] **Kaggle** – Work on dataset *(Rashmih)*
- [x] **anomoloy_scan.py** – Find large jumps in data before feeding to models *(Kushal, Rashmith)*
- [x] **models/** – Write all model classes 
    - [x] `lstm_model.py` – Implement LSTM model *(Kushal)*
    - [x] `auto_encoder.py` – Implement Auto Encoder model *(Kushal)* 
- [x] **scripts/** – Write pace scrum script *(Kushal)*
- [x] **pumpdumpsters/** – Verify our evaluation scripts *Ronak*
- [ ] **write checkin** – Write our checkin proposal *Together*

## Future To-Do
- [ ] `lstm_model.py` – Implement Hybrid CNN model *(Together)*
- [ ] **data_cleaning.py** – Clean data if necessary *(Ronak)*
- [ ] **feature_learning.py** – Implement feature learning if necessary *(Together)*

## Notes 3/22
* Pumpdumpsters is our evaluation python module. We need to go through and make each evaluation metric
* Wrote a skeleton script for auto encoder, but we need to go make better (lots of hardcoding!)
* Read over lstm model, may potentially need work
* Need a dataset. Right now we are locally using a test.csv
* Added pace support, but we may end up using elsewhere.
* Added sandbox.py, but we want to eventually remove it once we have all the code working.
