import os
import pandas as pd
from config import DATA_RAW

def verify_manual_download():
    dataset_path = os.path.join(DATA_RAW, "amazon_reviews.csv")
    
    if os.path.exists(dataset_path):
        file_size = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"✅ File found: {dataset_path}")
        print(f"✅ File size: {file_size:.2f} MB")
        
        try:
            df = pd.read_csv(dataset_path, nrows=5)
            print(f"✅ Dataset readable: {df.shape[1]} columns")
            print("Columns:", df.columns.tolist())
            print("\nSample data:")
            print(df.head(2))
            return True
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
    else:
        print("❌ File not found. Please download manually.")
        return False

if __name__ == "__main__":
    verify_manual_download()
