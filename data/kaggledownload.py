import os
import json
from kaggle import KaggleApi
# Define the correct path to kaggle.json inside your project
KAGGLE_JSON_PATH = 'UPDATE KAGGLE PATH'  # Update 

def setup_kaggle_credentials():
    """Load Kaggle API credentials and set them as environment variables."""
    if not os.path.exists(KAGGLE_JSON_PATH):
        raise FileNotFoundError(f"Kaggle credentials not found at {KAGGLE_JSON_PATH}. Please check the path.")
    with open(KAGGLE_JSON_PATH, "r") as f:
        creds = json.load(f)
    # Set API credentials as environment variables
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

def download_dataset():
    """Download the Kaggle dataset using environment variables."""
    setup_kaggle_credentials()

    # Initialize and authenticate the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Define the dataset identifier (owner/dataset-name)
    dataset = 'sudalairajkumar/cryptocurrencypricehistory'
    
    # Define the target directory for downloading the dataset
    target_dir = './raw/'
    
    # Download and unzip the dataset into the target directory
    api.dataset_download_files(dataset, path=target_dir, unzip=True)
    print(f"Dataset downloaded and extracted successfully into {target_dir}.")

if __name__ == '__main__':
    download_dataset()