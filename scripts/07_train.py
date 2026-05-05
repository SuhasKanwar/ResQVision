"""
Train a cyclone track prediction model.

Usage:
    python 07_train.py --model cnn_lstm --data_dir ../data --epochs 100

Models: tabular_lstm | cnn_mlp | cnn_lstm | convlstm | transformer
"""

import argparse
import os
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import importlib

def _lazy(alias, fname):
    spec = importlib.util.spec_from_file_location(
        alias, Path(__file__).parent / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ds  = _lazy('_ds',  '05_dataset.py')
_mdl = _lazy('_mdl', '06_models.py')
get_dataloaders = _ds.get_dataloaders
get_model       = _mdl.get_model
count_params    = _mdl.count_params


# ── Haversine MAE in km ───────────────────────────────────────────────────

def haversine_km(pred: torch.Tensor, target: torch.Tensor,
                 ref_lat: torch.Tensor, ref_lon: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, 2) Δlat/Δlon displacements
    ref_lat/ref_lon: (B,) absolute position at t0
    Returns: (B,) distance in km
    """
    pred_lat   = (ref_lat + pred[:, 0]).deg2rad()
    pred_lon   = (ref_lon + pred[:, 1]).deg2rad()
    true_lat   = (ref_lat + target[:, 0]).deg2rad()
    true_lon   = (ref_lon + target[:, 1]).deg2rad()
    dlat = true_lat - pred_lat
    dlon = true_lon - pred_lon
    a = torch.sin(dlat / 2) ** 2 + \
        torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(dlon / 2) ** 2
    return 6371.0 * 2 * torch.asin(torch.sqrt(a.clamp(0, 1)))


def haversine_mae(pred, target, tab):
    """Compute mean Haversine MAE in km from a batch."""
    # tab: (B, 4, 4) — last step (t0) cols are [lat, lon, wind, pres]
    ref_lat = tab[:, -1, 0].float()
    ref_lon = tab[:, -1, 1].float()
    return haversine_km(pred.float(), target.float(), ref_lat, ref_lon).mean().item()


# ── Training / evaluation loops ───────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss, total_mae, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, tab, target in loader:
            imgs   = imgs.to(device, non_blocking=True)
            tab    = tab.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(imgs, tab)
            loss = criterion(pred, target)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            B = imgs.size(0)
            total_loss += loss.item() * B
            total_mae  += haversine_mae(pred.detach(), target, tab) * B
            n += B

    return total_loss / n, total_mae / n


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      default='cnn_lstm',
                   choices=['tabular_lstm','cnn_mlp','cnn_lstm','convlstm','transformer'])
    p.add_argument('--data_dir',   default='../data')
    p.add_argument('--ckpt_dir',   default='../checkpoints')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience',   type=int, default=15)
    p.add_argument('--num_workers',type=int, default=8)
    p.add_argument('--shm_dir',    default='/dev/shm')
    p.add_argument('--compile',    action='store_true',
                   help='torch.compile() the model (PyTorch 2.0+)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] device={device}  model={args.model}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, _, scaler = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shm_dir=args.shm_dir,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = get_model(args.model).to(device)
    print(f"[train] parameters: {count_params(model):,}")
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("[train] torch.compile() enabled")

    # ── Optimiser & scheduler ─────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler_amp = None

    # ── Training loop ─────────────────────────────────────────────────
    best_val_mae = float('inf')
    patience_ctr = 0
    history = []
    ckpt_path = os.path.join(args.ckpt_dir, f'{args.model}_best.pt')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_mae = run_epoch(model, train_loader, criterion,
                                    optimizer, None, device, train=True)
        va_loss, va_mae = run_epoch(model, val_loader, criterion,
                                    None, None, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"Ep {epoch:03d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_mae={tr_mae:.1f} km  "
              f"val_loss={va_loss:.4f}  val_mae={va_mae:.1f} km  "
              f"lr={lr_now:.2e}  {elapsed:.1f}s")

        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_mae': tr_mae,
                        'val_loss': va_loss, 'val_mae': va_mae})

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            patience_ctr = 0
            torch.save({'epoch': epoch, 'model': args.model,
                        'state_dict': model.state_dict(),
                        'val_mae_km': va_mae,
                        'scaler_mean': scaler.mean_.tolist(),
                        'scaler_std': scaler.scale_.tolist()},
                       ckpt_path)
            print(f"  ✓ saved best checkpoint  val_mae={va_mae:.2f} km")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[train] early stopping at epoch {epoch}")
                break

    # ── Save history ──────────────────────────────────────────────────
    hist_path = os.path.join(args.ckpt_dir, f'{args.model}_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[train] done.  best val MAE = {best_val_mae:.2f} km")
    print(f"[train] checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
