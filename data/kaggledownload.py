import os
import json
from kaggle import KaggleApi
# Define the correct path to kaggle.json inside your project
KAGGLE_JSON_PATH = os.path.abspath("data/kaggle.json")  # Update this if needed

def setup_kaggle_credentials():
    """Load Kaggle API credentials and set them as environment variables."""
    if not os.path.exists(KAGGLE_JSON_PATH):
        raise FileNotFoundError(f"Kaggle credentials not found at {KAGGLE_JSON_PATH}. Please check the path.")
    with open("data/kaggle.json", "r") as f:
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
    
    # Download and unzip the dataset into the current directory
    api.dataset_download_files(dataset, path='.', unzip=True)
    print("Dataset downloaded and extracted successfully.")

if __name__ == '__main__':
    download_dataset()