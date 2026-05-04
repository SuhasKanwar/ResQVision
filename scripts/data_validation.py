import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_CLEAN_DIR = DATA_DIR / "processed" / "cleaned"
REPORTS_DIR = DATA_DIR / "reports"

def validate_data():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "validation_report.txt"
    
    splits = ['train_clean.csv', 'val_clean.csv', 'test_clean.csv']
    total_samples = 0
    missing_images = 0
    nan_values = 0
    
    with open(report_path, 'w') as f:
        f.write("Data Validation Report\n")
        f.write("======================\n\n")
        
        for split in splits:
            file_path = PROC_CLEAN_DIR / split
            if not file_path.exists():
                f.write(f"Missing file: {split}\n")
                continue
                
            df = pd.read_csv(file_path)
            total_samples += len(df)
            f.write(f"Split: {split}\n")
            f.write(f"Samples: {len(df)}\n")
            
            nans = df.isna().sum().sum()
            nan_values += nans
            f.write(f"Missing values (NaNs): {nans}\n")
            
            img_cols = ['img_path_t0', 'img_path_tminus1', 'img_path_tminus2', 'img_path_tminus3']
            missing_in_split = 0
            for col in img_cols:
                if col in df.columns:
                    for img_path in df[col]:
                        if not Path(img_path).exists():
                            missing_in_split += 1
            
            missing_images += missing_in_split
            f.write(f"Missing image files referenced: {missing_in_split}\n")
            
            if len(df) > 0:
                f.write(f"Lat range: {df['target_lat'].min():.2f} to {df['target_lat'].max():.2f}\n")
                f.write(f"Lon range: {df['target_lon'].min():.2f} to {df['target_lon'].max():.2f}\n")
                
            f.write("\n")
            
        f.write("Summary\n")
        f.write("-------\n")
        f.write(f"Total sequences validated: {total_samples}\n")
        f.write(f"Total missing values: {nan_values}\n")
        f.write(f"Total missing image references: {missing_images}\n")
        
        if nan_values == 0 and missing_images == 0 and total_samples > 0:
            f.write("\nSTATUS: PASS\n")
            logging.info("Validation passed.")
        else:
            f.write("\nSTATUS: FAIL/WARNING\n")
            logging.warning("Validation found issues.")
            
    logging.info(f"Validation report saved to {report_path}")

if __name__ == "__main__":
    validate_data()
