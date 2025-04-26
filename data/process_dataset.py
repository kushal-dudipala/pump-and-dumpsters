import os
import pandas as pd

def merge_all_crypto_csvs():
    """
    Merges all cryptocurrency CSV files from data/raw/ into a single CSV file inside data/processed/.
    """
    input_dir = os.path.join(os.path.dirname(__file__), "raw")
    output_file = os.path.join(os.path.dirname(__file__), "processed", "merged_cryptos.csv")

    all_dfs = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv") and filename.startswith("coin_"):
            filepath = os.path.join(input_dir, filename)
            print(f"Loading {filename}...")
            df = pd.read_csv(filepath)

            coin_name = filename.replace("coin_", "").replace(".csv", "")
            if 'Symbol' not in df.columns:
                df['Symbol'] = coin_name

            all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    print(f"\nMerged {len(all_dfs)} files into {output_file}")
    print(f"Total rows: {len(merged_df)}")


merge_all_crypto_csvs()