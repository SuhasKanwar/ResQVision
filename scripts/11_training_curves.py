import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {
    'tabular_lstm': '#2196F3',
    'cnn_mlp':      '#F44336',
    'cnn_lstm':     '#FF9800',
    'convlstm':     '#4CAF50',
    'transformer':  '#9C27B0',
}
LABELS = {
    'tabular_lstm': 'Tabular LSTM',
    'cnn_mlp':      'CNN + MLP',
    'cnn_lstm':     'CNN + LSTM',
    'convlstm':     'ConvLSTM',
    'transformer':  'Transformer',
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for name, c in COLORS.items():
    p = Path('checkpoints') / f'{name}_history.json'
    if not p.exists():
        continue
    h   = json.load(open(p))
    ep  = [x['epoch']     for x in h]
    tr  = [x['train_mae'] for x in h]
    val = [x['val_mae']   for x in h]
    axes[0].plot(ep, tr,  color=c, lw=1.5, ls='--', alpha=0.6)
    axes[1].plot(ep, val, color=c, lw=1.8, label=LABELS[name])

for ax, title in zip(axes, ['Train MAE (km)', 'Validation MAE (km)']):
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE (km)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(30, 100)

axes[1].legend(fontsize=9)
plt.suptitle('Training Convergence — All Models', fontsize=13)
plt.tight_layout()
plt.savefig('checkpoints/fig0_training_curves.png', dpi=150)
print('Saved → checkpoints/fig0_training_curves.png')
