import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    # Define the path to your Kaggle API credentials file
    kaggle_json_path = os.path.expanduser("~/data/kaggle.json")
    
    # Check if the credentials file exists
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_json_path}. "
            "Please ensure your kaggle.json file is correctly placed."
        )
    
    # Initialize and authenticate the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Define the dataset identifier (owner/dataset-name)
    dataset = 'sudalairajkumar/cryptocurrencypricehistory'
    
    # Download and unzip the dataset into the current directory
    api.dataset_download_files(dataset, path='.', unzip=True)
    print("Dataset downloaded and extracted successfully.")

if __name__ == '__main__':
    download_dataset()