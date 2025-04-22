import os
import json
import subprocess
import sys

def check_kaggle_credentials():
    """Check if Kaggle credentials are properly set up."""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_json):
        print("✅ Kaggle credentials found.")
        return True
    else:
        print("❌ Kaggle credentials not found.")
        print("\nTo set up Kaggle credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section and click 'Create New API Token'")
        print("3. This will download a kaggle.json file")
        print(f"4. Create directory: mkdir -p {kaggle_dir}")
        print(f"5. Move the downloaded file: mv ~/Downloads/kaggle.json {kaggle_json}")
        print(f"6. Set permissions: chmod 600 {kaggle_json}")
        return False

def install_kaggle_if_needed():
    """Install the Kaggle API if not already installed."""
    try:
        import kaggle
        print("✅ Kaggle API already installed.")
    except ImportError:
        print("⚙️ Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✅ Kaggle API installed successfully.")

def download_dataset():
    """Download the Kaggle dataset."""
    # Install Kaggle if needed
    install_kaggle_if_needed()
    
    # Check credentials
    if not check_kaggle_credentials():
        print("\n❗ Automatic download failed due to missing credentials.")
        print("\n📌 ALTERNATIVE: Manual Download Instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory")
        print("2. Click the 'Download' button on the website")
        print("3. Create a 'raw' directory in this project: mkdir -p ./raw")
        print("4. Extract the downloaded zip file into the 'raw' directory")
        return
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    # Initialize and authenticate the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Define the dataset identifier (owner/dataset-name)
    dataset = 'sudalairajkumar/cryptocurrencypricehistory'
    
    # Define the target directory for downloading the dataset
    target_dir = './raw/'
    os.makedirs(target_dir, exist_ok=True)
    
    # Download and unzip the dataset into the target directory
    print(f"⚙️ Downloading dataset from Kaggle...")
    api.dataset_download_files(dataset, path=target_dir, unzip=True)
    print(f"✅ Dataset downloaded and extracted successfully into {target_dir}.")

if __name__ == '__main__':
    download_dataset()