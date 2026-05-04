import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_SEQ_DIR = DATA_DIR / "processed" / "sequences"
PROC_CLEAN_DIR = DATA_DIR / "processed" / "cleaned"
REPORTS_DIR = DATA_DIR / "reports"

def clean_data():
    PROC_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    samples_path = PROC_SEQ_DIR / "dataset_samples.csv"
    if not samples_path.exists():
        logging.error(f"{samples_path} not found.")
        return

    df = pd.read_csv(samples_path)
    logging.info(f"Loaded {len(df)} samples.")
    
    num_cols = []
    for t in ['t0', 'tminus1', 'tminus2', 'tminus3']:
        num_cols.extend([f'lat_{t}', f'lon_{t}', f'wind_{t}', f'pres_{t}'])
        
    print("\n--- Missing stats before cleaning ---")
    missing_before = df[num_cols].isna().sum()
    print(missing_before)
    
    df[num_cols] = df.groupby('storm_id')[num_cols].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    
    print("\n--- Missing stats after cleaning ---")
    missing_after = df[num_cols].isna().sum()
    print(missing_after)
    
    clean_samples_path = PROC_CLEAN_DIR / "dataset_samples_clean.csv"
    df.to_csv(clean_samples_path, index=False)
    logging.info(f"Cleaned dataset saved to {clean_samples_path}")
    
    for split in ['train.csv', 'val.csv', 'test.csv']:
        split_path = PROC_SEQ_DIR / split
        if split_path.exists():
            split_df = pd.read_csv(split_path)
            
            cleaned_split = pd.merge(
                split_df[['storm_id', 'target_timestamp']], 
                df, 
                on=['storm_id', 'target_timestamp'], 
                how='left'
            )
            
            clean_split_path = PROC_CLEAN_DIR / f"{split.split('.')[0]}_clean.csv"
            cleaned_split.to_csv(clean_split_path, index=False)
            logging.info(f"Cleaned {split} saved to {clean_split_path} with {len(cleaned_split)} samples.")
            
    with open(REPORTS_DIR / "cleaning_report.txt", "w") as f:
        f.write("Data Cleaning Report\n")
        f.write("====================\n\n")
        f.write("Missing Values BEFORE Cleaning:\n")
        f.write(missing_before.to_string())
        f.write("\n\nMissing Values AFTER Cleaning:\n")
        f.write(missing_after.to_string())
        f.write(f"\n\nGlobal Medians used for fallback imputation:\n")
        f.write(medians.to_string())

if __name__ == "__main__":
    clean_data()
