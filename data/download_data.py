"""
Dataset verification and download utility for Online Shopping Clickstream Dataset.
"""

import os
import sys
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "shopping2.csv")

def verify_dataset():
    print("=" * 70)
    print("ONLINE SHOPPING CLICKSTREAM DATASET VERIFICATION")
    print("=" * 70)
    
    if not os.path.exists(DATA_FILE):
        print(f"[!] Dataset file NOT found at: {DATA_FILE}")
        print("    Please place 'shopping2.csv' inside the 'data/' directory.")
        print("    Original dataset source: UCI Machine Learning Repository / E-commerce clickstream")
        return False
        
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"[+] Dataset successfully loaded!")
        print(f"    File path: {DATA_FILE}")
        print(f"    Rows: {df.shape[0]:,}")
        print(f"    Columns: {df.shape[1]}")
        print(f"    Columns list: {list(df.columns)}")
        print(f"    Missing values: {df.isnull().sum().sum()}")
        print(f"    Memory usage: {df.memory_usage().sum() / (1024 * 1024):.2f} MB")
        print("=" * 70)
        return True
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
