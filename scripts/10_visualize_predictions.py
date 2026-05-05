"""
Visualise model predictions overlaid on actual HURSAT-B1 IR satellite images.

For each selected test sample:
  - Shows the 4 input images (t-3, t-2, t-1, t0) as a strip
  - On the t0 image overlays:
      ★ white  = current storm centre (t0)
      ● green  = true next position (t+6h)
      ✕ colour = each model's predicted position

Usage:
    python 10_visualize_predictions.py --data_dir data --ckpt_dir checkpoints
    python 10_visualize_predictions.py --n_samples 12 --data_dir data --ckpt_dir checkpoints
"""

import argparse
import importlib
import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── lazy imports ──────────────────────────────────────────────────────────────
def _lazy(alias, fname):
    spec = importlib.util.spec_from_file_location(
        alias, Path(__file__).parent / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ds  = _lazy('_ds',  '05_dataset.py')
_mdl = _lazy('_mdl', '06_models.py')

ALL_MODELS = ['tabular_lstm', 'cnn_mlp', 'cnn_lstm', 'convlstm', 'transformer']
MODEL_COLORS = {
    'tabular_lstm': '#2196F3',
    'cnn_mlp':      '#F44336',
    'cnn_lstm':     '#FF9800',
    'convlstm':     '#4CAF50',
    'transformer':  '#9C27B0',
}
MODEL_LABELS = {
    'tabular_lstm': 'Tabular LSTM',
    'cnn_mlp':      'CNN + MLP',
    'cnn_lstm':     'CNN + LSTM',
    'convlstm':     'ConvLSTM',
    'transformer':  'Transformer',
}

# HURSAT-B1 image geometry:
# Native grid: 301×301 px at 0.07°/px  → spans ±10.535° from storm centre
# After resize to 128×128: 0.07 × (301/128) ≈ 0.1641°/px
DEG_PER_PX = 0.07 * (301 / 128)   # ≈ 0.1641°/pixel
IMG_CTR    = 64                     # centre pixel (0-indexed)


def deg_to_px(dlat, dlon):
    """Convert (Δlat, Δlon) displacement from storm centre to image pixel coords."""
    col = IMG_CTR + dlon / DEG_PER_PX
    row = IMG_CTR - dlat / DEG_PER_PX   # north = up = smaller row
    return col, row


def haversine_km(pred_lat, pred_lon, true_lat, true_lon):
    R = 6371.0
    r = np.pi / 180
    dlat = (true_lat - pred_lat) * r
    dlon = (true_lon - pred_lon) * r
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(pred_lat * r) * np.cos(true_lat * r) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _remap(path, data_dir):
    p = path.replace('\\', '/')
    m = re.search(r'(processed/.+)', p)
    return str(Path(data_dir) / m.group(1)) if m else path


# ── load model ────────────────────────────────────────────────────────────────

def load_model(name, ckpt_dir, device):
    ckpt_path = Path(ckpt_dir) / f'{name}_best.pt'
    if not ckpt_path.exists():
        return None, None
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _mdl.get_model(name).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    sc = StandardScaler()
    sc.mean_  = np.array(ckpt['scaler_mean'])
    sc.scale_ = np.array(ckpt['scaler_std'])
    return model, sc


# ── per-sample prediction ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_sample(models_dict, scalers_dict, row, img_cache, data_dir, device):
    """Run all models on a single CSV row and return dict of (pred_lat, pred_lon)."""
    img_cols  = ['img_path_tminus3','img_path_tminus2','img_path_tminus1','img_path_t0']
    tab_cols  = [f'{f}_{s}' for s in ['tminus3','tminus2','tminus1','t0']
                 for f in ['lat','lon','wind','pres']]

    # Raw images (128×128 each)
    imgs_raw = []
    for col in img_cols:
        p = _remap(row[col], data_dir)
        imgs_raw.append(np.nan_to_num(np.load(p), nan=0.0, posinf=1.0, neginf=0.0))
    imgs_np = np.stack(imgs_raw, axis=0).astype(np.float32)   # (4,128,128)

    results = {}
    for name, model in models_dict.items():
        sc  = scalers_dict[name]
        tab = sc.transform(row[tab_cols].values.reshape(1, -1)).reshape(4, 4)
        imgs_t = torch.from_numpy(imgs_np).unsqueeze(0).to(device)   # (1,4,128,128)
        tab_t  = torch.from_numpy(tab.astype(np.float32)).unsqueeze(0).to(device)
        pred   = model(imgs_t, tab_t).cpu().numpy()[0]                # (2,)

        # Unscale reference lat/lon
        lat_mean, lat_std = sc.mean_[0],  sc.scale_[0]
        lon_mean, lon_std = sc.mean_[1],  sc.scale_[1]
        ref_lat = row['lat_t0']
        ref_lon = row['lon_t0']

        results[name] = {
            'pred_lat': ref_lat + pred[0],
            'pred_lon': ref_lon + pred[1],
            'dlat':     pred[0],
            'dlon':     pred[1],
        }
    return imgs_np, results


# ── main figure: image strip + overlays ──────────────────────────────────────

def plot_sample(imgs_np, row, preds, ax_strip, ax_overlay):
    """
    ax_strip  : 1×4 axes showing the 4 input images
    ax_overlay: the t0 image with prediction markers
    """
    titles = ['t−18h', 't−12h', 't−6h', 't0']
    cmap   = 'inferno_r'

    for i, ax in enumerate(ax_strip):
        ax.imshow(imgs_np[i], cmap=cmap, vmin=0, vmax=1, origin='upper')
        ax.set_title(titles[i], fontsize=8, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 3:
            ax.spines[:].set_edgecolor('yellow')
            ax.spines[:].set_linewidth(2)
        else:
            for sp in ax.spines.values():
                sp.set_visible(False)

    # ── overlay on t0 ────────────────────────────────────────────────────
    ax_overlay.imshow(imgs_np[3], cmap=cmap, vmin=0, vmax=1, origin='upper')
    ax_overlay.set_xticks([]); ax_overlay.set_yticks([])

    # Storm centre (t0) — always at image centre
    ax_overlay.plot(IMG_CTR, IMG_CTR, '*', color='white', ms=14,
                    markeredgecolor='black', markeredgewidth=0.8, zorder=10,
                    label='t0 centre')

    # True next position
    true_dlat = row['target_lat'] - row['lat_t0']
    true_dlon = row['target_lon'] - row['lon_t0']
    tc, tr = deg_to_px(true_dlat, true_dlon)
    ax_overlay.plot(tc, tr, 'o', color='limegreen', ms=12,
                    markeredgecolor='black', markeredgewidth=0.8, zorder=10,
                    label='True t+6h')
    # Arrow: centre → true
    ax_overlay.annotate('', xy=(tc, tr), xytext=(IMG_CTR, IMG_CTR),
                        arrowprops=dict(arrowstyle='->', color='limegreen',
                                        lw=1.8), zorder=9)

    # Model predictions
    for name, p in preds.items():
        pc, pr = deg_to_px(p['dlat'], p['dlon'])
        err_km = haversine_km(p['pred_lat'], p['pred_lon'],
                              row['target_lat'], row['target_lon'])
        ax_overlay.plot(pc, pr, 'X', color=MODEL_COLORS[name], ms=10,
                        markeredgecolor='black', markeredgewidth=0.5,
                        zorder=11)
        ax_overlay.annotate('', xy=(pc, pr), xytext=(IMG_CTR, IMG_CTR),
                            arrowprops=dict(arrowstyle='->', lw=1.2,
                                            color=MODEL_COLORS[name],
                                            alpha=0.7), zorder=8)

    # Degree scale bar (10° = 10/DEG_PER_PX px)
    scale_px = 5 / DEG_PER_PX   # 5° bar
    ax_overlay.plot([5, 5 + scale_px], [122, 122], '-', color='white', lw=2)
    ax_overlay.text(5 + scale_px / 2, 119, '5°', color='white',
                    fontsize=6, ha='center')

    # Storm info
    info = (f"{row['storm_id']}\n"
            f"Wind: {row['wind_t0']:.0f} kt  "
            f"Pres: {row['pres_t0']:.0f} mb")
    ax_overlay.set_title(info, fontsize=7, pad=3, color='white',
                         backgroundcolor='#00000088')


def make_legend(models_in_use):
    handles = [
        Line2D([0],[0], marker='*', color='white', ls='', ms=10,
               markeredgecolor='black', label='t0 centre'),
        Line2D([0],[0], marker='o', color='limegreen', ls='', ms=10,
               markeredgecolor='black', label='True t+6h'),
    ]
    for m in models_in_use:
        handles.append(
            Line2D([0],[0], marker='X', color=MODEL_COLORS[m], ls='', ms=9,
                   markeredgecolor='black', label=MODEL_LABELS[m]))
    return handles


# ── figure A: full grid (n_samples rows × 5 cols) ────────────────────────────

def plot_full_grid(samples, models_dict, scalers_dict, data_dir, device, out_path):
    """
    Each row = one sample.
    Cols: [t−18h image] [t−12h] [t−6h] [t0 plain] [t0 + predictions]
    """
    n  = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(5 * 5, n * 3.5),
                             facecolor='#111111')
    fig.subplots_adjust(wspace=0.04, hspace=0.3)

    for r, (_, row) in enumerate(samples.iterrows()):
        imgs_np, preds = predict_sample(
            models_dict, scalers_dict, row, None, data_dir, device)
        row_axes = axes[r] if n > 1 else axes
        plot_sample(imgs_np, row, preds,
                    ax_strip=row_axes[:4], ax_overlay=row_axes[4])

    # Column headers
    for ci, title in enumerate(['t−18h', 't−12h', 't−6h', 't0 (input)', 't0 + Predictions']):
        (axes[0] if n > 1 else axes)[ci].set_title(
            title, fontsize=9, color='white', pad=4)

    # Legend
    handles = make_legend(list(models_dict.keys()))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=9, framealpha=0.3,
               labelcolor='white', facecolor='#222222',
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('HURSAT-B1 IR Imagery — Predicted vs True Cyclone Displacement (6h)',
                 fontsize=13, color='white', y=1.005)

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#111111')
    plt.close()
    print(f'  → {out_path.name}')


# ── figure B: compact 2-col layout (image | zoomed overlay) ──────────────────

def plot_compact(samples, models_dict, scalers_dict, data_dir, device, out_path):
    """
    Left panel  : 4-frame strip
    Right panel : t0 with predictions + error table
    """
    n   = len(samples)
    fig = plt.figure(figsize=(14, n * 3.8), facecolor='#111111')

    for r, (_, row) in enumerate(samples.iterrows()):
        imgs_np, preds = predict_sample(
            models_dict, scalers_dict, row, None, data_dir, device)

        # 4 small image axes
        for c in range(4):
            ax = fig.add_axes([0.02 + c * 0.115, 1 - (r+1)/n + 0.01,
                               0.11, 0.85/n])
            ax.imshow(imgs_np[c], cmap='inferno_r', vmin=0, vmax=1,
                      origin='upper')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(['t−18h','t−12h','t−6h','t0'][c],
                             color='white', fontsize=8)

        # Large overlay
        ax_big = fig.add_axes([0.49, 1 - (r+1)/n + 0.01, 0.32, 0.85/n])
        ax_big.imshow(imgs_np[3], cmap='inferno_r', vmin=0, vmax=1,
                      origin='upper')
        ax_big.set_xticks([]); ax_big.set_yticks([])

        ax_big.plot(IMG_CTR, IMG_CTR, '*', color='white', ms=14,
                    markeredgecolor='black', zorder=10)
        true_dlat = row['target_lat'] - row['lat_t0']
        true_dlon = row['target_lon'] - row['lon_t0']
        tc, tr = deg_to_px(true_dlat, true_dlon)
        ax_big.plot(tc, tr, 'o', color='limegreen', ms=12,
                    markeredgecolor='black', zorder=10)
        ax_big.annotate('', xy=(tc, tr), xytext=(IMG_CTR, IMG_CTR),
                        arrowprops=dict(arrowstyle='->', color='limegreen',
                                        lw=2), zorder=9)
        for name, p in preds.items():
            pc, pr = deg_to_px(p['dlat'], p['dlon'])
            ax_big.plot(pc, pr, 'X', color=MODEL_COLORS[name], ms=11,
                        markeredgecolor='black', markeredgewidth=0.5, zorder=11)
            ax_big.annotate('', xy=(pc, pr), xytext=(IMG_CTR, IMG_CTR),
                            arrowprops=dict(arrowstyle='->', lw=1.3,
                                            color=MODEL_COLORS[name],
                                            alpha=0.75), zorder=8)

        title = (f"{row['storm_id']}  "
                 f"Wind {row['wind_t0']:.0f}kt  Pres {row['pres_t0']:.0f}mb")
        ax_big.set_title(title, fontsize=7, color='white',
                         backgroundcolor='#00000099', pad=2)

        # Error table on the right
        ax_tbl = fig.add_axes([0.83, 1 - (r+1)/n + 0.05, 0.16, 0.75/n])
        ax_tbl.set_facecolor('#1a1a1a')
        ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
        for sp in ax_tbl.spines.values():
            sp.set_color('grey')

        y = 0.92
        ax_tbl.text(0.5, y, 'Error (km)', ha='center', va='top',
                    color='white', fontsize=7, fontweight='bold',
                    transform=ax_tbl.transAxes)
        true_err_label = f"True: ({true_dlat:+.2f}°, {true_dlon:+.2f}°)"
        ax_tbl.text(0.05, y - 0.12, true_err_label, ha='left', va='top',
                    color='limegreen', fontsize=6,
                    transform=ax_tbl.transAxes)
        y -= 0.28
        for name, p in preds.items():
            km = haversine_km(p['pred_lat'], p['pred_lon'],
                              row['target_lat'], row['target_lon'])
            label = f"{MODEL_LABELS[name][:12]}: {km:.1f}"
            ax_tbl.text(0.05, y, label, ha='left', va='top',
                        color=MODEL_COLORS[name], fontsize=6.5,
                        transform=ax_tbl.transAxes)
            y -= 0.16

    handles = make_legend(list(models_dict.keys()))
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=8, framealpha=0.4, labelcolor='white',
               facecolor='#222222', bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        'IR Satellite Imagery with Predicted vs True 6-Hour Cyclone Displacement',
        fontsize=12, color='white', y=1.008)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f'  → {out_path.name}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',    default='../data')
    ap.add_argument('--ckpt_dir',    default='../checkpoints')
    ap.add_argument('--n_samples',   type=int, default=8)
    ap.add_argument('--seed',        type=int, default=42)
    args = ap.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.ckpt_dir)
    cleaned = Path(args.data_dir) / 'processed' / 'cleaned'

    # Load all models
    models_dict  = {}
    scalers_dict = {}
    for name in ALL_MODELS:
        m, sc = load_model(name, args.ckpt_dir, device)
        if m is not None:
            models_dict[name]  = m
            scalers_dict[name] = sc
            print(f'Loaded {name}')

    if not models_dict:
        print('No checkpoints found.')
        return

    # Sample test rows — pick variety: different storms, intensities
    test_df = pd.read_csv(str(cleaned / 'test_clean.csv'))
    # One sample per unique storm (capped at n_samples)
    storms  = test_df['storm_id'].unique()
    rng     = np.random.default_rng(args.seed)
    chosen  = rng.choice(storms, size=min(args.n_samples, len(storms)), replace=False)
    samples = []
    for sid in chosen:
        sub = test_df[test_df['storm_id'] == sid]
        # Pick the row with highest wind speed for visual interest
        samples.append(sub.loc[sub['wind_t0'].idxmax()])
    samples = pd.DataFrame(samples).reset_index(drop=True)

    print(f'\nGenerating visualisations for {len(samples)} storms …')

    plot_full_grid(samples, models_dict, scalers_dict,
                   args.data_dir, device,
                   out_dir / 'fig7_image_predictions_grid.png')

    plot_compact(samples, models_dict, scalers_dict,
                 args.data_dir, device,
                 out_dir / 'fig8_image_predictions_compact.png')

    print('\nDone. Figures saved to', out_dir)


if __name__ == '__main__':
    main()
