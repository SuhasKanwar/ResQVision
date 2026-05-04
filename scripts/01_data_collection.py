import os
import pandas as pd
import numpy as np
import xarray as xr
import requests
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_IBTRACS_DIR = DATA_DIR / "raw" / "ibtracs"
RAW_HURSAT_DIR = DATA_DIR / "raw" / "hursat"
INTERIM_DIR = DATA_DIR / "interim"
PROC_IMAGES_DIR = DATA_DIR / "processed" / "images"
PROC_SEQ_DIR = DATA_DIR / "processed" / "sequences"

MAX_STORMS_TO_DOWNLOAD = None  

def create_dirs():
    for d in [RAW_IBTRACS_DIR, RAW_HURSAT_DIR, INTERIM_DIR, PROC_IMAGES_DIR, PROC_SEQ_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def download_ibtracs():
    logging.info("Starting IBTrACS download...")
    url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NI.list.v04r00.csv"
    dest_path = RAW_IBTRACS_DIR / "ibtracs.NI.csv"
    
    if not dest_path.exists():
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(dest_path, "wb") as f, tqdm(
                desc="Downloading IBTrACS",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            logging.info("IBTrACS download complete.")
        except Exception as e:
            logging.error(f"Failed to download IBTrACS: {e}")
            raise
    else:
        logging.info("IBTrACS data already exists.")
        
    return dest_path

def clean_ibtracs(file_path):
    logging.info("Cleaning IBTrACS data...")
    df = pd.read_csv(file_path, low_memory=False, skiprows=[1])
    
    df = df[df['BASIN'] == 'NI'].copy()
    
    cols_to_keep = ['SID', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'USA_PRES', 'BASIN', 'SEASON']
    df = df[cols_to_keep].copy()
    
    df.rename(columns={
        'SID': 'storm_id',
        'NAME': 'storm_name',
        'ISO_TIME': 'timestamp',
        'LAT': 'latitude',
        'LON': 'longitude',
        'USA_WIND': 'wind_speed',
        'USA_PRES': 'pressure',
        'BASIN': 'basin',
        'SEASON': 'season'
    }, inplace=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for col in ['latitude', 'longitude', 'wind_speed', 'pressure']:
        df[col] = pd.to_numeric(df[col].replace(' ', np.nan), errors='coerce')
        
    df = df[df['timestamp'].dt.hour.isin([0, 6, 12, 18])]
    
    df.sort_values(['storm_id', 'timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    clean_path = INTERIM_DIR / "ibtracs_clean.csv"
    df.to_csv(clean_path, index=False)
    logging.info(f"Cleaned IBTrACS saved to {clean_path}")
    return df

def download_hursat(storms_df):
    logging.info("Downloading/Mocking HURSAT data...")
    unique_storms = storms_df['storm_id'].unique()
    
    if MAX_STORMS_TO_DOWNLOAD is not None:
        unique_storms = unique_storms[-MAX_STORMS_TO_DOWNLOAD:]
        logging.info(f"Limiting to {MAX_STORMS_TO_DOWNLOAD} storms for demonstration.")

    storms_to_process = storms_df[storms_df['storm_id'].isin(unique_storms)]
    
    for _, row in tqdm(storms_to_process.iterrows(), total=len(storms_to_process), desc="Generating HURSAT Data"):
        storm_id = row['storm_id']
        year = row['timestamp'].year
        dt_str = row['timestamp'].strftime('%Y%m%d%H')
        
        storm_dir = RAW_HURSAT_DIR / str(year) / storm_id
        storm_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = storm_dir / f"hursat_v6_{storm_id}_{dt_str}.nc"
        
        if not filepath.exists():
            mock_data = np.random.rand(1, 150, 150).astype(np.float32)
            ds = xr.Dataset(
                {
                    "IRWIN": (["time", "lat", "lon"], mock_data)
                },
                coords={
                    "time": [row['timestamp']],
                    "lat": np.linspace(row['latitude']-5, row['latitude']+5, 150),
                    "lon": np.linspace(row['longitude']-5, row['longitude']+5, 150)
                },
                attrs={
                    "storm_name": row['storm_name'],
                    "satellite": "MockSat-1"
                }
            )
            ds.to_netcdf(filepath)
            ds.close()

def parse_hursat_metadata():
    logging.info("Parsing HURSAT metadata...")
    records = []
    
    nc_files = list(RAW_HURSAT_DIR.rglob('*.nc'))
    for filepath in tqdm(nc_files, desc="Parsing metadata"):
        try:
            with xr.open_dataset(filepath) as ds:
                time_val = pd.to_datetime(ds.time.values[0])
                records.append({
                    'filepath': str(filepath),
                    'storm_name': ds.attrs.get('storm_name', 'UNKNOWN'),
                    'timestamp': time_val,
                    'satellite': ds.attrs.get('satellite', 'UNKNOWN'),
                    'image_shape': ds.IRWIN.shape
                })
        except Exception as e:
            logging.warning(f"Failed to parse {filepath}: {e}")
            
    df = pd.DataFrame(records)
    if not df.empty:
        metadata_path = INTERIM_DIR / "hursat_metadata.csv"
        df.to_csv(metadata_path, index=False)
        logging.info(f"Saved metadata to {metadata_path}")
    return df

def match_timestamps(ibtracs_df, hursat_df):
    logging.info("Matching timestamps...")
    if hursat_df.empty:
        logging.error("No HURSAT data available to match.")
        return pd.DataFrame()
        
    i_df = ibtracs_df.sort_values('timestamp')
    h_df = hursat_df.sort_values('timestamp')
    
    i_df['timestamp'] = pd.to_datetime(i_df['timestamp']).astype('datetime64[ns]')
    h_df['timestamp'] = pd.to_datetime(h_df['timestamp']).astype('datetime64[ns]')
    
    matched = pd.merge_asof(
        i_df, 
        h_df, 
        on='timestamp', 
        by='storm_name', 
        direction='nearest', 
        tolerance=pd.Timedelta('3 hours')
    )
    
    matched = matched.dropna(subset=['filepath']).copy()
    matched_path = INTERIM_DIR / "storm_time_matches.csv"
    matched.to_csv(matched_path, index=False)
    logging.info(f"Matched {len(matched)} records.")
    return matched

def preprocess_images(matched_df):
    logging.info("Preprocessing images...")
    target_shape = (128, 128)
    processed_records = []
    
    for idx, row in tqdm(matched_df.iterrows(), total=len(matched_df), desc="Processing images"):
        fp = row['filepath']
        try:
            with xr.open_dataset(fp) as ds:
                img_array = ds.IRWIN.values[0]
                
                img_min = np.nanmin(img_array)
                img_max = np.nanmax(img_array)
                if img_max > img_min:
                    img_norm = (img_array - img_min) / (img_max - img_min)
                else:
                    img_norm = img_array
                    
                img_pil = Image.fromarray(img_norm)
                img_resized = img_pil.resize(target_shape, Image.Resampling.BILINEAR)
                img_final = np.array(img_resized)
                
                save_name = f"img_{row['storm_id']}_{row['timestamp'].strftime('%Y%m%d%H')}.npy"
                save_path = PROC_IMAGES_DIR / save_name
                np.save(save_path, img_final)
                
                processed_records.append({
                    'storm_id': row['storm_id'],
                    'timestamp': row['timestamp'],
                    'processed_img_path': str(save_path)
                })
        except Exception as e:
            logging.warning(f"Error processing {fp}: {e}")
            
    return pd.DataFrame(processed_records)

def build_sequences(matched_df, proc_df):
    logging.info("Building sequences...")
    if matched_df.empty or proc_df.empty:
        return pd.DataFrame()
        
    df = pd.merge(matched_df, proc_df, on=['storm_id', 'timestamp'])
    df = df.sort_values(['storm_id', 'timestamp'])
    
    seq_length = 4
    samples = []
    
    grouped = df.groupby('storm_id')
    for storm_id, group in tqdm(grouped, total=len(grouped), desc="Building sequences"):
        group = group.reset_index(drop=True)
        if len(group) < seq_length + 1:
            continue
            
        for i in range(len(group) - seq_length):
            window = group.iloc[i:i+seq_length]
            target = group.iloc[i+seq_length]
            
            time_diffs = group['timestamp'].iloc[i:i+seq_length+1].diff()[1:]
            if not all(time_diffs == pd.Timedelta(hours=6)):
                continue
                
            img_paths = window['processed_img_path'].tolist()
            
            sample = {
                'storm_id': storm_id,
                'target_timestamp': target['timestamp'],
                'target_lat': target['latitude'],
                'target_lon': target['longitude'],
                'img_path_t0': img_paths[3],
                'img_path_tminus1': img_paths[2],
                'img_path_tminus2': img_paths[1],
                'img_path_tminus3': img_paths[0],
                'lat_t0': window.iloc[3]['latitude'],
                'lon_t0': window.iloc[3]['longitude'],
                'wind_t0': window.iloc[3]['wind_speed'],
                'pres_t0': window.iloc[3]['pressure'],
                'lat_tminus1': window.iloc[2]['latitude'],
                'lon_tminus1': window.iloc[2]['longitude'],
                'wind_tminus1': window.iloc[2]['wind_speed'],
                'pres_tminus1': window.iloc[2]['pressure'],
                'lat_tminus2': window.iloc[1]['latitude'],
                'lon_tminus2': window.iloc[1]['longitude'],
                'wind_tminus2': window.iloc[1]['wind_speed'],
                'pres_tminus2': window.iloc[1]['pressure'],
                'lat_tminus3': window.iloc[0]['latitude'],
                'lon_tminus3': window.iloc[0]['longitude'],
                'wind_tminus3': window.iloc[0]['wind_speed'],
                'pres_tminus3': window.iloc[0]['pressure']
            }
            samples.append(sample)
            
    samples_df = pd.DataFrame(samples)
    if not samples_df.empty:
        samples_path = PROC_SEQ_DIR / "dataset_samples.csv"
        samples_df.to_csv(samples_path, index=False)
        logging.info(f"Generated {len(samples_df)} sequences.")
    return samples_df

def split_by_storm(samples_df):
    logging.info("Splitting train/val/test...")
    if samples_df.empty:
        logging.warning("No samples to split.")
        return 0, 0, 0, 0
        
    unique_storms = samples_df['storm_id'].unique()
    
    if len(unique_storms) < 3:
        logging.warning("Not enough storms for a proper 70/15/15 split. Using all for train.")
        samples_df.to_csv(PROC_SEQ_DIR / "train.csv", index=False)
        pd.DataFrame(columns=samples_df.columns).to_csv(PROC_SEQ_DIR / "val.csv", index=False)
        pd.DataFrame(columns=samples_df.columns).to_csv(PROC_SEQ_DIR / "test.csv", index=False)
        return len(unique_storms), len(samples_df), 0, 0
        
    train_storms, temp_storms = train_test_split(unique_storms, test_size=0.3, random_state=42)
    val_storms, test_storms = train_test_split(temp_storms, test_size=0.5, random_state=42)
    
    train_df = samples_df[samples_df['storm_id'].isin(train_storms)]
    val_df = samples_df[samples_df['storm_id'].isin(val_storms)]
    test_df = samples_df[samples_df['storm_id'].isin(test_storms)]
    
    train_df.to_csv(PROC_SEQ_DIR / "train.csv", index=False)
    val_df.to_csv(PROC_SEQ_DIR / "val.csv", index=False)
    test_df.to_csv(PROC_SEQ_DIR / "test.csv", index=False)
    
    return len(unique_storms), len(train_df), len(val_df), len(test_df)

def main():
    create_dirs()
    
    ibtracs_path = download_ibtracs()
    ibtracs_clean_df = clean_ibtracs(ibtracs_path)
    
    download_hursat(ibtracs_clean_df)
    hursat_meta_df = parse_hursat_metadata()
    
    matched_df = match_timestamps(ibtracs_clean_df, hursat_meta_df)
    
    proc_df = preprocess_images(matched_df)
    
    samples_df = build_sequences(matched_df, proc_df)
    
    total_storms, train_size, val_size, test_size = split_by_storm(samples_df)
    
    print("\n" + "="*40)
    print("CYCLONE PATH PREDICTION - DATA PIPELINE SUMMARY")
    print("="*40)
    print(f"Total Storms Used       : {total_storms}")
    print(f"Total Sequence Samples  : {len(samples_df)}")
    print(f"Image Final Shape       : (128, 128)")
    print(f"Training Samples        : {train_size}")
    print(f"Validation Samples      : {val_size}")
    print(f"Testing Samples         : {test_size}")
    print("="*40)

if __name__ == "__main__":
    main()