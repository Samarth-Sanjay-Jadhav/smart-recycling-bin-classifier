"""Script to download TrashNet dataset."""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile

def download_trashnet():
    """Download TrashNet dataset."""
    
    dataset_url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    dataset_path = Path("data/dataset-resized.zip")
    extract_path = Path("data")
    
    print("Starting download of TrashNet dataset...")
    print(f"URL: {dataset_url}\n")
    
    try:
        # Create data directory
        Path("data").mkdir(exist_ok=True)
        
        # Download
        print(f"Downloading to {dataset_path}...")
        urllib.request.urlretrieve(dataset_url, dataset_path)
        print("✅ Download complete!")
        
        # Extract
        print(f"Extracting to {extract_path}...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extraction complete!")
        
        # Cleanup
        os.remove(dataset_path)
        print("✅ Cleanup complete!")
        
        print("\n✅ Dataset ready at: data/dataset-resized/")
        print("Next step: python models/train_classifier.py")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_trashnet()