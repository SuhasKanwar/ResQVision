import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Shared CNN Encoder (used by CNN+MLP and CNN+LSTM)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Lightweight CNN that maps (B, 1, 128, 128) → (B, out_dim)."""
    def __init__(self, in_channels: int = 1, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            self._block(in_channels, 32),   # → (32, 64, 64)
            self._block(32,  64),            # → (64, 32, 32)
            self._block(64,  128),           # → (128, 16, 16)
            self._block(128, 256),           # → (256, 8, 8)
            nn.AdaptiveAvgPool2d(1),         # → (256, 1, 1)
            nn.Flatten(),                    # → (256,)
        )
        self.proj = nn.Linear(256, out_dim) if out_dim != 256 else nn.Identity()

    @staticmethod
    def _block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.proj(self.net(x))


# ---------------------------------------------------------------------------
# Model 1 — Tabular LSTM
# ---------------------------------------------------------------------------

class TabularLSTM(nn.Module):
    """
    Pure meteorological baseline.
    Input:  tab (B, 4, 4) — (lat, lon, wind, pres) at t-3…t0
    Output: (B, 2)        — (Δlat, Δlon)
    """
    def __init__(self, input_dim: int = 4, hidden: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, imgs, tab):
        # imgs unused — tabular-only model
        _, (h, _) = self.lstm(tab)          # h: (num_layers, B, hidden)
        return self.head(h[-1])


# ---------------------------------------------------------------------------
# Model 2 — CNN + MLP (no temporal modelling)
# ---------------------------------------------------------------------------

class CNN_MLP(nn.Module):
    """
    All 4 images stacked as channels, spatial features + tabular → MLP.
    Input:  imgs (B, 4, 128, 128), tab (B, 4, 4)
    Output: (B, 2)
    """
    def __init__(self, img_feat_dim: int = 256, tab_dim: int = 16,
                 dropout: float = 0.3):
        super().__init__()
        self.cnn  = CNNEncoder(in_channels=4, out_dim=img_feat_dim)
        fused_dim = img_feat_dim + tab_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, imgs, tab):
        img_feat = self.cnn(imgs)                          # (B, 256)
        tab_flat = tab.flatten(1)                          # (B, 16)
        return self.head(torch.cat([img_feat, tab_flat], dim=1))


# ---------------------------------------------------------------------------
# Model 3 — CNN + LSTM (primary model)
# ---------------------------------------------------------------------------

class CNN_LSTM(nn.Module):
    """
    Shared CNN encodes each timestep; LSTM models the sequence.
    Input:  imgs (B, 4, 128, 128), tab (B, 4, 4)
    Output: (B, 2)
    """
    def __init__(self, img_feat_dim: int = 256, hidden: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.cnn  = CNNEncoder(in_channels=1, out_dim=img_feat_dim)
        lstm_in   = img_feat_dim + 4        # CNN features + per-step tabular
        self.lstm = nn.LSTM(lstm_in, hidden, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, imgs, tab):
        B, T, H, W = imgs.shape                            # (B, 4, 128, 128)
        # Encode each timestep independently with shared CNN
        img_feats = self.cnn(imgs.view(B * T, 1, H, W))   # (B*4, 256)
        img_feats = img_feats.view(B, T, -1)               # (B, 4, 256)
        # Concatenate per-step tabular
        seq = torch.cat([img_feats, tab], dim=-1)          # (B, 4, 260)
        _, (h, _) = self.lstm(seq)
        return self.head(h[-1])


# ---------------------------------------------------------------------------
# ConvLSTM cell (for Model 4)
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(in_channels + hidden_channels,
                               4 * hidden_channels, kernel_size, padding=pad)

    def forward(self, x, states):
        h, c = states
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, batch, height, width, device):
        return (torch.zeros(batch, self.hidden_channels, height, width, device=device),
                torch.zeros(batch, self.hidden_channels, height, width, device=device))


# ---------------------------------------------------------------------------
# Model 4 — ConvLSTM
# ---------------------------------------------------------------------------

class ConvLSTMModel(nn.Module):
    """
    Spatiotemporal recurrence directly in feature space.
    Input:  imgs (B, 4, 128, 128), tab (B, 4, 4)
    Output: (B, 2)
    """
    def __init__(self, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.cell1 = ConvLSTMCell(1,      hidden, kernel_size=3)
        self.cell2 = ConvLSTMCell(hidden, hidden, kernel_size=3)
        self.post  = nn.Sequential(
            nn.Conv2d(hidden, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),   # → 64*4*4 = 1024
        )
        self.head = nn.Sequential(
            nn.Linear(1024 + 16, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, imgs, tab):
        B, T, H, W = imgs.shape
        h1, c1 = self.cell1.init_hidden(B, H, W, imgs.device)
        h2, c2 = self.cell2.init_hidden(B, H, W, imgs.device)
        for t in range(T):
            x = imgs[:, t:t+1]                            # (B, 1, H, W)
            h1, c1 = self.cell1(x,  (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
        spatial = self.post(h2)                            # (B, 1024)
        tab_flat = tab.flatten(1)                          # (B, 16)
        return self.head(torch.cat([spatial, tab_flat], dim=1))


# ---------------------------------------------------------------------------
# Model 5 — Patch Transformer
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 16, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, patch_size, stride=patch_size)
        # 128/16 = 8 → 8×8 = 64 patches per image

    def forward(self, x):
        # x: (B, 1, 128, 128) → (B, embed_dim, 8, 8) → (B, 64, embed_dim)
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)               # (B, 64, embed_dim)


class PatchTransformer(nn.Module):
    """
    Each image → 64 patch tokens; temporal + spatial position encoding;
    4-head self-attention × 3 layers over all 4×64=256 tokens.
    Input:  imgs (B, 4, 128, 128), tab (B, 4, 4)
    Output: (B, 2)
    """
    def __init__(self, patch_size: int = 16, embed_dim: int = 128,
                 nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        n_patches = (128 // patch_size) ** 2              # 64

        # Learnable temporal + spatial position embeddings
        self.temp_embed    = nn.Embedding(4, embed_dim)
        self.spatial_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.norm        = nn.LayerNorm(embed_dim)

        self.tab_proj = nn.Linear(16, 32)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 32, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),             nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, imgs, tab):
        B, T, H, W = imgs.shape
        patches_all = []
        for t in range(T):
            p = self.patch_embed(imgs[:, t:t+1])           # (B, 64, embed)
            p = p + self.spatial_embed
            p = p + self.temp_embed(
                torch.full((B,), t, dtype=torch.long, device=imgs.device)
            ).unsqueeze(1)
            patches_all.append(p)
        tokens = torch.cat(patches_all, dim=1)             # (B, 256, embed)

        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, embed)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, 257, embed)
        tokens = self.transformer(tokens)
        cls_out = self.norm(tokens[:, 0])                  # (B, embed)

        tab_feat = self.tab_proj(tab.flatten(1))           # (B, 32)
        return self.head(torch.cat([cls_out, tab_feat], dim=1))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODELS = {
    'tabular_lstm': TabularLSTM,
    'cnn_mlp':      CNN_MLP,
    'cnn_lstm':     CNN_LSTM,
    'convlstm':     ConvLSTMModel,
    'transformer':  PatchTransformer,
}


def get_model(name: str) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS)}")
    return MODELS[name]()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
