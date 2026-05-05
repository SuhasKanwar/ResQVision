"""
Print a comparison table of all trained models.

Usage:
    python 09_compare.py --ckpt_dir ../checkpoints
"""

import argparse
import json
import importlib
from pathlib import Path

ALL_MODELS = ['tabular_lstm', 'cnn_mlp', 'cnn_lstm', 'convlstm', 'transformer']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', default='../checkpoints')
    args = p.parse_args()

    ckpt_dir  = Path(args.ckpt_dir)
    eval_json = ckpt_dir / 'eval_results.json'

    if eval_json.exists():
        with open(eval_json) as f:
            results = json.load(f)
    else:
        results = {}

    # Also pull best val MAE from checkpoints
    import torch
    ckpt_info = {}
    for name in ALL_MODELS:
        ckpt_path = ckpt_dir / f'{name}_best.pt'
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location='cpu')
            ckpt_info[name] = {'epoch': ck.get('epoch', '?'),
                               'val_mae': ck.get('val_mae_km', '?')}

    # ── Table ────────────────────────────────────────────────────────
    header = f"{'Model':<18} {'Params':>8}  {'Best Ep':>7}  {'Val MAE':>9}  " \
             f"{'Test MAE':>9}  {'Median':>8}  {'P90':>8}  {'RMSE':>8}"
    print("\n" + "=" * len(header))
    print("CYCLONE TRACK PREDICTION — MODEL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Count params
    spec = importlib.util.spec_from_file_location(
        '_mdl', Path(__file__).parent / '06_models.py')
    mdl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mdl)

    for name in ALL_MODELS:
        try:
            params = f"{mdl.count_params(mdl.get_model(name)):,}"
        except Exception:
            params = "?"

        ep      = ckpt_info.get(name, {}).get('epoch', '—')
        val_mae = ckpt_info.get(name, {}).get('val_mae', None)
        val_str = f"{val_mae:.2f} km" if isinstance(val_mae, float) else "—"

        r = results.get(name, {})
        test_mae = f"{r['mae_km']:.2f} km"   if 'mae_km'    in r else "—"
        median   = f"{r['median_km']:.2f} km" if 'median_km' in r else "—"
        p90      = f"{r['p90_km']:.2f} km"   if 'p90_km'    in r else "—"
        rmse     = f"{r['rmse_km']:.2f} km"  if 'rmse_km'   in r else "—"

        print(f"{name:<18} {params:>8}  {str(ep):>7}  {val_str:>9}  "
              f"{test_mae:>9}  {median:>8}  {p90:>8}  {rmse:>8}")

    print("=" * len(header))

    # Training curves
    print("\nTraining history files:")
    for name in ALL_MODELS:
        hist = ckpt_dir / f'{name}_history.json'
        if hist.exists():
            with open(hist) as f:
                h = json.load(f)
            best = min(h, key=lambda x: x['val_mae'])
            print(f"  {name:<18} converged ep {best['epoch']:>3}  "
                  f"val_mae={best['val_mae']:.2f} km")


if __name__ == '__main__':
    main()
