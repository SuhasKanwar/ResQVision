import os
import re
import io
import tarfile
import pandas as pd
import numpy as np
import xarray as xr
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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

HURSAT_BASE_URL = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"
HURSAT_MIN_YEAR = 1978  # HURSAT-B1 v6 coverage starts 1978
HURSAT_MAX_YEAR = 2015  # HURSAT-B1 v6 coverage ends 2015
DOWNLOAD_WORKERS = 16   # parallel connections for listings + downloads

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

def _fetch_year_storm_map(year):
    """Fetch HURSAT year listing. Returns (year, {storm_id: tar_filename}, base_url)."""
    url = f"{HURSAT_BASE_URL}/{year}/"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            entries = re.findall(r'href="(HURSAT_b1_v06_([^_]+)_[^"]+\.tar\.gz)"', resp.text)
            return year, {sid: fname for fname, sid in entries}, url
    except requests.RequestException:
        pass
    return year, {}, url

def _download_and_extract(tar_url, storm_dir):
    """Download a HURSAT storm tar.gz and extract all .nc files into storm_dir."""
    resp = requests.get(tar_url, timeout=300)
    resp.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode='r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.nc'):
                member.name = Path(member.name).name
                tar.extract(member, path=storm_dir)

def download_hursat(storms_df):
    logging.info("Downloading HURSAT-B1 data from NCEI (coverage: 1978–2015)...")

    df = storms_df.copy()
    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    df = df[(df['year'] >= HURSAT_MIN_YEAR) & (df['year'] <= HURSAT_MAX_YEAR)]

    if df.empty:
        logging.warning(
            "No NI basin storms found within HURSAT-B1 coverage (up to 2015). "
            "IBTrACS data exists but all storms fall outside the satellite archive range."
        )
        return

    unique_storms = df['storm_id'].unique()
    if MAX_STORMS_TO_DOWNLOAD is not None:
        unique_storms = unique_storms[-MAX_STORMS_TO_DOWNLOAD:]
        logging.info(f"Limiting to {MAX_STORMS_TO_DOWNLOAD} storms.")

    storm_year = {sid: int(df[df['storm_id'] == sid]['year'].iloc[0]) for sid in unique_storms}

    # Skip storms whose directories already have .nc files
    storms_to_fetch = [
        sid for sid in unique_storms
        if not list((RAW_HURSAT_DIR / str(storm_year[sid]) / sid).glob('*.nc'))
    ]
    already_done = len(unique_storms) - len(storms_to_fetch)
    if already_done:
        logging.info(f"Skipping {already_done} storms already on disk.")
    if not storms_to_fetch:
        logging.info("All HURSAT-B1 data already on disk.")
        return

    # Phase 1: fetch one listing per year (much fewer requests than per-storm)
    years_needed = sorted({storm_year[sid] for sid in storms_to_fetch})
    logging.info(f"Fetching listings for {len(years_needed)} years...")
    year_maps, year_urls = {}, {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for year, storm_map, base_url in pool.map(_fetch_year_storm_map, years_needed):
            year_maps[year] = storm_map
            year_urls[year] = base_url

    # Phase 2: match IBTrACS SIDs to HURSAT tar.gz filenames
    download_tasks = []
    not_in_archive = []
    for sid in storms_to_fetch:
        year = storm_year[sid]
        tar_fname = year_maps.get(year, {}).get(sid)
        if tar_fname:
            storm_dir = RAW_HURSAT_DIR / str(year) / sid
            storm_dir.mkdir(parents=True, exist_ok=True)
            download_tasks.append((year_urls[year] + tar_fname, storm_dir))
        else:
            not_in_archive.append(sid)

    logging.info(f"Matched {len(download_tasks)} storms, {len(not_in_archive)} not in archive.")

    # Phase 3: download + extract tar.gz files in parallel
    downloaded, errors = 0, 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_download_and_extract, url, d): d for url, d in download_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading & extracting"):
            storm_dir = futures[future]
            try:
                future.result()
                downloaded += 1
            except Exception as e:
                errors += 1
                logging.warning(f"Failed {storm_dir.name}: {e}")

    logging.info(f"HURSAT-B1 complete: {downloaded} storms extracted, {errors} errors, {len(not_in_archive)} not in archive.")
    if not_in_archive:
        logging.info(f"Not in archive: {not_in_archive[:10]}{'...' if len(not_in_archive) > 10 else ''}")

def _parse_one_nc(filepath):
    # Extract timestamp from filename — zero file I/O, avoids HDF5 thread issues.
    # Real HURSAT filename: {storm_id}.{sub}.{YYYY}.{MM}.{DD}.{HHMM}.*.hursat-b1.v06.nc
    try:
        parts = Path(filepath).stem.split('.')
        # parts[2]=YYYY, [3]=MM, [4]=DD, [5]=HHMM
        ts = pd.Timestamp(
            year=int(parts[2]), month=int(parts[3]), day=int(parts[4]),
            hour=int(parts[5][:2]), minute=int(parts[5][2:])
        )
        return {
            'filepath': str(filepath),
            'storm_id': Path(filepath).parent.name,
            'timestamp': ts,
            'satellite': 'HURSAT-B1',
            'image_shape': (1, 301, 301)
        }
    except Exception as e:
        logging.warning(f"Failed to parse {filepath}: {e}")
        return None

def parse_hursat_metadata():
    logging.info("Parsing HURSAT metadata...")
    # Only scan directories within the HURSAT coverage window to skip old mock files
    nc_files = [
        fp for fp in RAW_HURSAT_DIR.rglob('*.nc')
        if fp.parent.parent.name.isdigit()
        and HURSAT_MIN_YEAR <= int(fp.parent.parent.name) <= HURSAT_MAX_YEAR
    ]
    logging.info(f"Found {len(nc_files)} .nc files within {HURSAT_MIN_YEAR}–{HURSAT_MAX_YEAR}.")
    records = []

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_parse_one_nc, fp): fp for fp in nc_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing metadata"):
            result = future.result()
            if result:
                records.append(result)

    df = pd.DataFrame(records)
    if not df.empty:
        metadata_path = INTERIM_DIR / "hursat_metadata.csv"
        df.to_csv(metadata_path, index=False)
        logging.info(f"Saved metadata for {len(df)} files to {metadata_path}")
    return df

def match_timestamps(ibtracs_df, hursat_df):
    logging.info("Matching timestamps...")
    if hursat_df.empty:
        logging.error("No HURSAT data available to match.")
        return pd.DataFrame()

    i_df = ibtracs_df.sort_values('timestamp').copy()
    h_df = hursat_df[['storm_id', 'timestamp', 'filepath', 'satellite']].sort_values('timestamp').copy()

    i_df['timestamp'] = pd.to_datetime(i_df['timestamp']).astype('datetime64[ns]')
    h_df['timestamp'] = pd.to_datetime(h_df['timestamp']).astype('datetime64[ns]')

    matched = pd.merge_asof(
        i_df,
        h_df,
        on='timestamp',
        by='storm_id',
        direction='nearest',
        tolerance=pd.Timedelta('3 hours')
    )

    matched = matched.dropna(subset=['filepath']).copy()
    matched_path = INTERIM_DIR / "storm_time_matches.csv"
    matched.to_csv(matched_path, index=False)
    logging.info(f"Matched {len(matched)} records.")
    return matched

def _process_one_image(row, target_shape=(128, 128)):
    try:
        with xr.open_dataset(row['filepath']) as ds:
            img_array = ds['IRWIN'].values[0]  # dim: (htime, lat, lon)
            img_min, img_max = np.nanmin(img_array), np.nanmax(img_array)
            img_norm = ((img_array - img_min) / (img_max - img_min)).astype(np.float32) \
                if img_max > img_min else np.zeros_like(img_array, dtype=np.float32)
            img_final = np.array(
                Image.fromarray(img_norm).resize(target_shape, Image.Resampling.BILINEAR)
            )
            save_path = PROC_IMAGES_DIR / f"img_{row['storm_id']}_{row['timestamp'].strftime('%Y%m%d%H')}.npy"
            np.save(save_path, img_final)
            return {'storm_id': row['storm_id'], 'timestamp': row['timestamp'], 'processed_img_path': str(save_path)}
    except Exception as e:
        logging.warning(f"Error processing {row['filepath']}: {e}")
        return None

def preprocess_images(matched_df):
    logging.info("Preprocessing images...")
    rows = [row._asdict() for row in matched_df.itertuples(index=False)]
    records = []

    # ProcessPoolExecutor avoids HDF5 global lock that serialises ThreadPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process_one_image, row): row for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result:
                records.append(result)

    return pd.DataFrame(records)

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