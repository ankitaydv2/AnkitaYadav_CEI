"""
Dataset Downloader for Store Item Demand Forecasting
=====================================================
Downloads train.csv and test.csv from a public GitHub mirror of the
Kaggle Store Item Demand Forecasting Challenge dataset.

Source: https://github.com/worshipneo/Store-Item-Demand-Forecasting-1

Files are saved to: data/
    - train.csv  (913,000 rows — 5 years of daily sales)
    - test.csv   (45,000 rows — 3 months to forecast)

Usage:
    python download_data.py
"""

import os
import urllib.request

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully downloaded {output_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Try main branch if master fails
        if "/master/" in url:
            new_url = url.replace("/master/", "/main/")
            print(f"Retrying with main branch: {new_url}")
            try:
                urllib.request.urlretrieve(new_url, output_path)
                print(f"Successfully downloaded {output_path} from main")
            except Exception as e2:
                print(f"Failed again: {e2}")
                raise e2
        else:
            raise e

def main():
    base_url = "https://raw.githubusercontent.com/worshipneo/Store-Item-Demand-Forecasting-1/master"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    files = ["train.csv", "test.csv"]
    for file in files:
        url = f"{base_url}/{file}"
        output_path = os.path.join(data_dir, file)
        if not os.path.exists(output_path):
            download_file(url, output_path)
        else:
            print(f"{file} already exists, skipping download.")
            
    # Check if we can get sample_submission.csv
    url = f"{base_url}/sample_submission.csv"
    output_path = os.path.join(data_dir, "sample_submission.csv")
    if not os.path.exists(output_path):
        try:
            download_file(url, output_path)
        except Exception:
            print("Failed to download sample_submission.csv, we will generate it later.")
    else:
        print("sample_submission.csv already exists, skipping download.")

if __name__ == "__main__":
    main()
