"""
Evaluate all trained models and generate visualisations.

Usage:
    python 08_evaluate.py --model all  --data_dir data --ckpt_dir checkpoints
    python 08_evaluate.py --model cnn_lstm --data_dir data --ckpt_dir checkpoints
"""

import argparse
import json
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
import importlib

# ── lazy imports ─────────────────────────────────────────────────────────────
def _lazy(alias, fname):
    spec = importlib.util.spec_from_file_location(
        alias, Path(__file__).parent / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ds  = _lazy('_ds',  '05_dataset.py')
_mdl = _lazy('_mdl', '06_models.py')

ALL_MODELS   = ['tabular_lstm', 'cnn_mlp', 'cnn_lstm', 'convlstm', 'transformer']
MODEL_COLORS = {
    'tabular_lstm': '#2196F3',   # blue
    'cnn_mlp':      '#F44336',   # red
    'cnn_lstm':     '#FF9800',   # orange
    'convlstm':     '#4CAF50',   # green
    'transformer':  '#9C27B0',   # purple
}
MODEL_LABELS = {
    'tabular_lstm': 'Tabular LSTM',
    'cnn_mlp':      'CNN + MLP',
    'cnn_lstm':     'CNN + LSTM',
    'convlstm':     'ConvLSTM',
    'transformer':  'Transformer',
}


# ── metrics ──────────────────────────────────────────────────────────────────

def haversine_km_np(pred_lat, pred_lon, true_lat, true_lon):
    R = 6371.0
    r = np.pi / 180
    dlat = (true_lat - pred_lat) * r
    dlon = (true_lon - pred_lon) * r
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(pred_lat * r) * np.cos(true_lat * r) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_metrics(pred_lat, pred_lon, true_lat, true_lon):
    errs = haversine_km_np(pred_lat, pred_lon, true_lat, true_lon)
    return {
        'mae_km':    float(np.mean(errs)),
        'median_km': float(np.median(errs)),
        'p90_km':    float(np.percentile(errs, 90)),
        'rmse_km':   float(np.sqrt(np.mean(errs ** 2))),
        'mae_lat':   float(np.mean(np.abs(pred_lat - true_lat))),
        'mae_lon':   float(np.mean(np.abs(pred_lon - true_lon))),
        'errors':    errs.tolist(),
    }


# ── model loading & inference ─────────────────────────────────────────────────

def load_model(model_name, ckpt_dir, device):
    ckpt_path = Path(ckpt_dir) / f'{model_name}_best.pt'
    if not ckpt_path.exists():
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _mdl.get_model(model_name).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_  = np.array(ckpt['scaler_mean'])
    scaler.scale_ = np.array(ckpt['scaler_std'])
    return model, scaler


@torch.no_grad()
def run_inference(model, loader, device):
    """Returns arrays: pred_lat, pred_lon, true_lat, true_lon, ref_lat, ref_lon."""
    P_lat, P_lon, T_lat, T_lon, R_lat, R_lon = [], [], [], [], [], []
    for imgs, tab, target in loader:
        imgs   = imgs.to(device, non_blocking=True)
        tab    = tab.to(device, non_blocking=True)
        pred   = model(imgs, tab).cpu().numpy()
        t      = target.numpy()
        rl     = tab[:, -1, 0].cpu().numpy()   # lat_t0 (normalised)
        rlon   = tab[:, -1, 1].cpu().numpy()   # lon_t0 (normalised)
        P_lat.append(pred[:, 0]); P_lon.append(pred[:, 1])
        T_lat.append(t[:, 0]);    T_lon.append(t[:, 1])
        R_lat.append(rl);         R_lon.append(rlon)
    return (np.concatenate(P_lat), np.concatenate(P_lon),
            np.concatenate(T_lat), np.concatenate(T_lon),
            np.concatenate(R_lat), np.concatenate(R_lon))


# ── figure 1: per-model error distribution ───────────────────────────────────

def plot_error_distribution(all_errors: dict, out_dir: Path):
    """Overlapping KDE-style histograms for all models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, errs in all_errors.items():
        ax.hist(errs, bins=60, range=(0, 300), density=True,
                alpha=0.45, color=MODEL_COLORS[name],
                label=f"{MODEL_LABELS[name]}  (MAE={np.mean(errs):.1f} km)")
        ax.axvline(np.mean(errs), color=MODEL_COLORS[name], lw=1.5, ls='--')
    ax.set_xlabel('Track Error (km)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('6-Hour Track Error Distribution — All Models (Test Set)', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 300)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig1_error_distributions.png', dpi=150)
    plt.close()
    print('  → fig1_error_distributions.png')


# ── figure 2: boxplot comparison ─────────────────────────────────────────────

def plot_boxplot_comparison(all_errors: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    data   = [all_errors[m] for m in ALL_MODELS if m in all_errors]
    labels = [MODEL_LABELS[m] for m in ALL_MODELS if m in all_errors]
    colors = [MODEL_COLORS[m] for m in ALL_MODELS if m in all_errors]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='white', lw=2),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker='o', ms=2, alpha=0.3))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Track Error (km)', fontsize=12)
    ax.set_title('Track Error Comparison — Test Set', fontsize=13)
    ax.set_ylim(0, 400)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig2_boxplot_comparison.png', dpi=150)
    plt.close()
    print('  → fig2_boxplot_comparison.png')


# ── figure 3: MAE bar chart ───────────────────────────────────────────────────

def plot_mae_bar(results: dict, out_dir: Path):
    names  = [m for m in ALL_MODELS if m in results]
    maes   = [results[m]['mae_km'] for m in names]
    colors = [MODEL_COLORS[m] for m in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(names)), maes, color=colors, edgecolor='white', width=0.55)
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f'{v:.2f} km', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in names], fontsize=10)
    ax.set_ylabel('Test MAE (km)', fontsize=12)
    ax.set_title('6-Hour Track Prediction MAE — Test Set', fontsize=13)
    ax.set_ylim(0, max(maes) * 1.15)
    ax.axhline(47.75, color='grey', lw=1, ls=':', label='Tabular LSTM val MAE (ref)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig3_mae_bar.png', dpi=150)
    plt.close()
    print('  → fig3_mae_bar.png')


# ── figure 4: sample storm tracks with prediction arrows ─────────────────────

def plot_storm_tracks(test_csv, all_preds: dict, scaler, out_dir: Path, n_storms=6):
    """
    For n_storms randomly chosen test storms, plot:
    - Actual track (blue line with dots)
    - At each t0 position: arrows to true_next (green) and each model's predicted_next (coloured)
    """
    df = pd.read_csv(test_csv)
    storms = df['storm_id'].unique()
    rng    = np.random.default_rng(42)
    chosen = rng.choice(storms, size=min(n_storms, len(storms)), replace=False)

    n_cols = 3
    n_rows = int(np.ceil(len(chosen) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5.5, n_rows * 4.5))
    axes = axes.flatten()

    for ax_i, storm_id in enumerate(chosen):
        ax  = axes[ax_i]
        sub = df[df['storm_id'] == storm_id].reset_index(drop=True)

        # Actual track: lat_t0 / lon_t0 sequence
        lats = sub['lat_t0'].values
        lons = sub['lon_t0'].values
        ax.plot(lons, lats, 'o-', color='steelblue', lw=1.5, ms=4,
                label='Track (t0)', zorder=3)

        # True next positions
        true_lat = sub['target_lat'].values
        true_lon = sub['target_lon'].values
        ax.scatter(true_lon, true_lat, marker='*', s=60, color='black',
                   zorder=4, label='True t+6h')

        # Per-model predicted arrows
        for model_name, pred_data in all_preds.items():
            pred_lat_arr, pred_lon_arr, _, _, _, _, indices = pred_data
            mask = indices  # bool mask for this storm
            if not np.any(mask):
                continue
            p_lat = pred_lat_arr[mask]
            p_lon = pred_lon_arr[mask]
            t_lat = true_lat
            t_lon = true_lon
            # Draw arrows from t0 → predicted
            for j in range(min(len(p_lat), len(lats))):
                ax.annotate('', xy=(p_lon[j], p_lat[j]),
                            xytext=(lons[j], lats[j]),
                            arrowprops=dict(arrowstyle='->', lw=1.2,
                                            color=MODEL_COLORS[model_name],
                                            alpha=0.8))

        ax.set_title(f'{storm_id}', fontsize=8)
        ax.set_xlabel('Lon', fontsize=7); ax.set_ylabel('Lat', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    # Remove empty panels
    for ax_i in range(len(chosen), len(axes)):
        axes[ax_i].set_visible(False)

    # Legend
    handles = [Line2D([0], [0], color='steelblue', lw=1.5, label='Track (t0)'),
               Line2D([0], [0], marker='*', color='black', ls='', ms=8, label='True t+6h')]
    for m in all_preds:
        handles.append(mpatches.Patch(color=MODEL_COLORS[m],
                                      label=MODEL_LABELS[m]))
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Sample Storm Tracks — Predicted vs True Next Position (6h)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig4_storm_tracks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  → fig4_storm_tracks.png')


# ── figure 5: NIO scatter map ─────────────────────────────────────────────────

def plot_nio_map(all_preds: dict, out_dir: Path):
    """Scatter map of true vs predicted positions (best model = ConvLSTM or first available)."""
    best = 'convlstm' if 'convlstm' in all_preds else list(all_preds.keys())[0]
    pred_lat, pred_lon, true_lat, true_lon = (
        all_preds[best][0], all_preds[best][1],
        all_preds[best][2], all_preds[best][3])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#e8f4f8')

    # NIO rough coastline box
    ax.set_xlim(40, 110); ax.set_ylim(-5, 35)
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)

    ax.scatter(true_lon, true_lat, s=12, alpha=0.4, color='steelblue',
               label='True t+6h position', zorder=3)
    ax.scatter(pred_lon, pred_lat, s=12, alpha=0.4, color='tomato',
               marker='^', label=f'{MODEL_LABELS[best]} prediction', zorder=3)

    # Error lines for a random sample
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(pred_lat), size=min(200, len(pred_lat)), replace=False)
    for i in idx:
        ax.plot([true_lon[i], pred_lon[i]], [true_lat[i], pred_lat[i]],
                color='grey', lw=0.4, alpha=0.3)

    mae = np.mean(haversine_km_np(pred_lat, pred_lon, true_lat, true_lon))
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'NIO Test Set — True vs Predicted Positions ({MODEL_LABELS[best]}, MAE={mae:.1f} km)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig5_nio_map.png', dpi=150)
    plt.close()
    print('  → fig5_nio_map.png')


# ── figure 6: pred vs true scatter per model ─────────────────────────────────

def plot_pred_vs_true(all_preds: dict, out_dir: Path):
    """2×3 grid: for each model, scatter of predicted Δlat/Δlon vs true."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (name, data) in enumerate(all_preds.items()):
        ax = axes[i]
        pred_lat, pred_lon, true_lat, true_lon = data[0], data[1], data[2], data[3]
        # Use displacements: ref_lat from data[4], ref_lon from data[5]
        ref_lat, ref_lon = data[4], data[5]
        d_pred = pred_lat - ref_lat
        d_true = true_lat - ref_lat

        ax.scatter(d_true, d_pred, s=8, alpha=0.3, color=MODEL_COLORS[name])
        lim = max(np.abs(d_true).max(), np.abs(d_pred).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='Perfect')
        ax.set_xlabel('True Δlat (°)', fontsize=9)
        ax.set_ylabel('Predicted Δlat (°)', fontsize=9)
        mae = np.mean(haversine_km_np(pred_lat, pred_lon, true_lat, true_lon))
        ax.set_title(f'{MODEL_LABELS[name]}\nMAE = {mae:.2f} km', fontsize=10)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.2)

    axes[-1].set_visible(False)
    fig.suptitle('Predicted vs True Δlat — All Models (Test Set)', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig6_pred_vs_true.png', dpi=150)
    plt.close()
    print('  → fig6_pred_vs_true.png')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',       default='all')
    ap.add_argument('--data_dir',    default='../data')
    ap.add_argument('--ckpt_dir',    default='../checkpoints')
    ap.add_argument('--batch_size',  type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--shm_dir',     default='/dev/shm')
    args = ap.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models   = ALL_MODELS if args.model == 'all' else [args.model]
    out_dir  = Path(args.ckpt_dir)
    cleaned  = Path(args.data_dir) / 'processed' / 'cleaned'
    test_csv = str(cleaned / 'test_clean.csv')

    # Build shared image cache once
    dfs = [pd.read_csv(str(cleaned / f'{s}_clean.csv')) for s in ('train','val','test')]
    img_cache = _ds.build_image_cache(dfs, args.data_dir, args.shm_dir)

    all_results  = {}
    all_errors   = {}
    all_preds    = {}   # name → (pred_lat, pred_lon, true_lat, true_lon, ref_lat, ref_lon, storm_mask_fn)

    for model_name in models:
        print(f'\n── {model_name} ──')
        model, scaler = load_model(model_name, args.ckpt_dir, device)
        if model is None:
            print(f'  no checkpoint, skipping')
            continue

        from torch.utils.data import DataLoader
        test_ds = _ds.CycloneDataset(test_csv, args.data_dir, img_cache,
                                     scaler, augment=False)
        loader  = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

        p_dlat, p_dlon, t_dlat, t_dlon, r_lat, r_lon = run_inference(
            model, loader, device)

        # Convert scaler-normalised ref to real degrees
        lat_mean, lat_std = scaler.mean_[0], scaler.scale_[0]
        lon_mean, lon_std = scaler.mean_[1], scaler.scale_[1]
        ref_lat_real = r_lat * lat_std + lat_mean
        ref_lon_real = r_lon * lon_std + lon_mean

        # The model outputs raw Δlat/Δlon but scaler was on absolute lat/lon.
        # Target was NOT scaled — it's raw degrees displacement.
        pred_lat = ref_lat_real + p_dlat
        pred_lon = ref_lon_real + p_dlon
        true_lat = ref_lat_real + t_dlat
        true_lon = ref_lon_real + t_dlon

        metrics = compute_metrics(pred_lat, pred_lon, true_lat, true_lon)
        all_results[model_name] = {k: v for k, v in metrics.items() if k != 'errors'}
        all_errors[model_name]  = np.array(metrics['errors'])

        print(f'  MAE    : {metrics["mae_km"]:.2f} km')
        print(f'  Median : {metrics["median_km"]:.2f} km')
        print(f'  P90    : {metrics["p90_km"]:.2f} km')
        print(f'  RMSE   : {metrics["rmse_km"]:.2f} km')

        # Build per-storm index arrays for track plots
        test_df  = pd.read_csv(test_csv).reset_index(drop=True)
        storms   = test_df['storm_id'].values
        all_preds[model_name] = (pred_lat, pred_lon, true_lat, true_lon,
                                 ref_lat_real, ref_lon_real, storms)

    if not all_results:
        print('No models evaluated.')
        return

    # ── Generate all figures ─────────────────────────────────────────────
    print('\nGenerating figures …')
    plot_error_distribution(all_errors, out_dir)
    plot_boxplot_comparison(all_errors, out_dir)
    plot_mae_bar(all_results, out_dir)
    plot_pred_vs_true(all_preds, out_dir)
    plot_nio_map(all_preds, out_dir)

    # Storm track plots (only if we have multiple models for comparison)
    if len(all_preds) > 1:
        # Build per-storm mask dict
        track_preds = {}
        test_df = pd.read_csv(test_csv).reset_index(drop=True)
        for m, data in all_preds.items():
            track_preds[m] = data

        # Simplified track plot using storm-level grouping
        _plot_track_grid(test_df, track_preds, out_dir)

    # Save JSON
    out_json = out_dir / 'eval_results.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved → {out_json}')
    print('All figures saved to', out_dir)


def _plot_track_grid(test_df, all_preds, out_dir, n_storms=6):
    """Plot n_storms from test set showing each model's prediction arrows."""
    storms = test_df['storm_id'].unique()
    rng    = np.random.default_rng(42)
    chosen = rng.choice(storms, size=min(n_storms, len(storms)), replace=False)

    n_cols = 3
    n_rows = int(np.ceil(len(chosen) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5.5, n_rows * 5))
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for ai, sid in enumerate(chosen):
        ax  = axes_flat[ai]
        sub = test_df[test_df['storm_id'] == sid].reset_index(drop=True)
        idx = sub.index.tolist()  # row indices in the full test_df

        # Build global index mapping: row in test_df → position in pred arrays
        global_indices = sub.index.tolist()

        lats = sub['lat_t0'].values
        lons = sub['lon_t0'].values
        t_lats = sub['target_lat'].values
        t_lons = sub['target_lon'].values

        # Actual track
        ax.plot(lons, lats, 'o-', color='steelblue', lw=1.8, ms=5,
                zorder=4, label='Track t0')
        ax.scatter(t_lons, t_lats, marker='*', s=80,
                   color='black', zorder=5, label='True t+6h')

        # Model predictions
        for m, data in all_preds.items():
            p_lat, p_lon = data[0], data[1]
            p_lat_s = p_lat[global_indices]
            p_lon_s = p_lon[global_indices]
            ax.scatter(p_lon_s, p_lat_s, marker='x', s=40,
                       color=MODEL_COLORS[m], zorder=5, alpha=0.8)
            for j in range(len(lats)):
                ax.annotate('', xy=(p_lon_s[j], p_lat_s[j]),
                            xytext=(lons[j], lats[j]),
                            arrowprops=dict(arrowstyle='->', lw=1.0,
                                            color=MODEL_COLORS[m], alpha=0.7))

        ax.set_title(f'{sid}', fontsize=8)
        ax.set_xlabel('Lon', fontsize=7)
        ax.set_ylabel('Lat', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

    for ai in range(len(chosen), len(axes_flat)):
        axes_flat[ai].set_visible(False)

    # Legend
    handles = [
        Line2D([0], [0], color='steelblue', marker='o', lw=1.5, label='Track t0'),
        Line2D([0], [0], marker='*', color='black', ls='', ms=10, label='True t+6h'),
    ]
    for m in all_preds:
        handles.append(Line2D([0], [0], marker='x', color=MODEL_COLORS[m],
                               ls='', ms=8, label=MODEL_LABELS[m]))
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Test Storm Tracks — All Models Predicted vs True Next Position',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig4_storm_tracks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  → fig4_storm_tracks.png')


if __name__ == '__main__':
    main()
