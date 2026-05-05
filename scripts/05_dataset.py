import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm

# Column order: oldest → newest (t-3, t-2, t-1, t0)
IMG_COLS   = ['img_path_tminus3', 'img_path_tminus2', 'img_path_tminus1', 'img_path_t0']
FEATS      = ['lat', 'lon', 'wind', 'pres']
STEPS      = ['tminus3', 'tminus2', 'tminus1', 't0']
TAB_COLS   = [f'{f}_{s}' for s in STEPS for f in FEATS]   # 16 features, (4, 4) when reshaped
TARGET_COLS = ['target_lat', 'target_lon']
REF_COLS    = ['lat_t0', 'lon_t0']
IMG_SIZE    = 128


def _remap(path: str, data_dir: str) -> str:
    """Remap a stored absolute path (Windows or Linux) to the local data_dir."""
    p = path.replace('\\', '/')
    m = re.search(r'(processed/.+)', p)
    if m:
        return str(Path(data_dir) / m.group(1))
    return path


def build_image_cache(dfs: list, data_dir: str, shm_dir: str = '/dev/shm') -> dict:
    """
    Load every unique .npy image into a single RAM array.
    All 4,795 images = ~315 MB — fits trivially.
    On Linux, forked DataLoader workers share this array via copy-on-write.
    Uses /dev/shm if available for maximum throughput.
    """
    all_paths = set()
    for df in dfs:
        for col in IMG_COLS:
            all_paths.update(_remap(p, data_dir) for p in df[col].tolist())

    paths = sorted(all_paths)
    n = len(paths)
    print(f"[dataset] loading {n} images ({n * IMG_SIZE * IMG_SIZE * 4 / 1e6:.1f} MB) into RAM …")

    shm_file = os.path.join(shm_dir, 'cyclone_imgs.npy')
    try:
        arr = np.memmap(shm_file, dtype=np.float32, mode='w+', shape=(n, IMG_SIZE, IMG_SIZE))
        print(f"[dataset] using /dev/shm memmap at {shm_file}")
    except Exception:
        arr = np.empty((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        print("[dataset] /dev/shm unavailable, using heap RAM")

    path_to_idx = {}
    for i, p in enumerate(tqdm(paths, desc="Loading images", ncols=80)):
        arr[i] = np.nan_to_num(np.load(p), nan=0.0, posinf=1.0, neginf=0.0)
        path_to_idx[p] = i

    return path_to_idx, arr


class CycloneDataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str,
                 img_cache: tuple,          # (path_to_idx dict, numpy array)
                 scaler: StandardScaler = None,
                 augment: bool = False):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.path_to_idx, self.img_arr = img_cache
        self.augment = augment

        # Tabular: shape (N, 4, 4) — (samples, timesteps, features)
        raw_tab = self.df[TAB_COLS].values.astype(np.float32)   # (N, 16)
        if scaler is not None:
            raw_tab = scaler.transform(raw_tab)
        self.tabular = raw_tab.reshape(-1, 4, 4)                 # (N, 4, 4)

        # Displacement target (Δlat, Δlon)
        targets = self.df[TARGET_COLS].values.astype(np.float32)
        refs    = self.df[REF_COLS].values.astype(np.float32)
        self.targets = targets - refs                             # (N, 2)

        # Image index lookup per sample: (N, 4)
        self.img_indices = np.array([
            [self.path_to_idx[_remap(row[c], data_dir)] for c in IMG_COLS]
            for _, row in self.df.iterrows()
        ], dtype=np.int32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Images: (4, 128, 128) float32
        imgs = self.img_arr[self.img_indices[idx]]          # (4, 128, 128)

        if self.augment:
            if np.random.rand() > 0.5:
                imgs = imgs[:, :, ::-1].copy()              # horizontal flip
            noise = np.random.randn(*self.tabular[idx].shape).astype(np.float32) * 0.01
            tab = self.tabular[idx] + noise
        else:
            tab = self.tabular[idx]

        return (
            torch.from_numpy(imgs),                         # (4, 128, 128)
            torch.from_numpy(tab),                          # (4, 4)
            torch.from_numpy(self.targets[idx]),            # (2,)
        )


def get_scaler(train_csv: str, data_dir: str) -> StandardScaler:
    df = pd.read_csv(train_csv)
    scaler = StandardScaler()
    scaler.fit(df[TAB_COLS].values.astype(np.float32))
    return scaler


def get_dataloaders(data_dir: str,
                    batch_size: int = 128,
                    num_workers: int = 8,
                    shm_dir: str = '/dev/shm') -> tuple:
    cleaned = Path(data_dir) / 'processed' / 'cleaned'
    train_csv = str(cleaned / 'train_clean.csv')
    val_csv   = str(cleaned / 'val_clean.csv')
    test_csv  = str(cleaned / 'test_clean.csv')

    dfs = [pd.read_csv(p) for p in (train_csv, val_csv, test_csv)]
    img_cache = build_image_cache(dfs, data_dir, shm_dir)
    scaler    = get_scaler(train_csv, data_dir)

    train_ds = CycloneDataset(train_csv, data_dir, img_cache, scaler, augment=True)
    val_ds   = CycloneDataset(val_csv,   data_dir, img_cache, scaler, augment=False)
    test_ds  = CycloneDataset(test_csv,  data_dir, img_cache, scaler, augment=False)

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, scaler
