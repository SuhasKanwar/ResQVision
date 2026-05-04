import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_CLEAN_DIR = DATA_DIR / "processed" / "cleaned"
REPORTS_DIR = DATA_DIR / "reports"

def perform_eda():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    samples_path = PROC_CLEAN_DIR / "dataset_samples_clean.csv"
    
    if not samples_path.exists():
        logging.error("Cleaned dataset samples not found.")
        return
        
    df = pd.read_csv(samples_path)
    
    plt.figure(figsize=(10, 8))
    for storm_id, group in df.groupby('storm_id'):
        plt.plot(group['target_lon'], group['target_lat'], marker='.', linestyle='-', alpha=0.5)
    plt.title('Cyclone Tracks (Target Lat vs Target Lon)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig(REPORTS_DIR / "cyclone_tracks.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='pres_t0', y='wind_t0', alpha=0.6)
    plt.title('Wind Speed vs Pressure at t0')
    plt.xlabel('Pressure (mb)')
    plt.ylabel('Wind Speed (knots)')
    plt.grid(True)
    plt.savefig(REPORTS_DIR / "wind_vs_pressure.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.histplot(df['wind_t0'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Wind Speeds at t0')
    plt.xlabel('Wind Speed (knots)')
    plt.savefig(REPORTS_DIR / "wind_distribution.png")
    plt.close()

    with open(REPORTS_DIR / "eda_summary.txt", "w") as f:
        f.write("Exploratory Data Analysis Summary\n")
        f.write("=================================\n")
        f.write(f"Total sequence samples: {len(df)}\n")
        f.write(f"Unique storms in sequences: {df['storm_id'].nunique()}\n")
        f.write(f"Average Wind Speed (t0): {df['wind_t0'].mean():.2f} knots\n")
        f.write(f"Average Pressure (t0): {df['pres_t0'].mean():.2f} mb\n")
        
        wind_missing = df['wind_t0'].isna().sum()
        pres_missing = df['pres_t0'].isna().sum()
        f.write(f"Missing Wind values (t0): {wind_missing}\n")
        f.write(f"Missing Pressure values (t0): {pres_missing}\n")

    logging.info(f"EDA plots and summary saved to {REPORTS_DIR}")

if __name__ == "__main__":
    perform_eda()