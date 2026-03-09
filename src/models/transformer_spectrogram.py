"""
Lightweight hybrid CNN-Transformer for spectrogram classification.
Uses a CNN stem to downsample, then applies self-attention on a small token sequence.
This avoids the O(n²) bottleneck of pure ViT on large spectrograms.
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class RFTransformer(nn.Module):
    """Hybrid CNN-Transformer for RF spectrogram classification.

    Architecture:
        1. CNN stem: 3 conv layers with pooling → reduces spatial dims by ~16x
           (257x511 → ~16x32 = ~512 tokens)
        2. Flatten spatial dims into token sequence
        3. Transformer encoder (2 layers of self-attention)
        4. CLS token → classification head

    This is much faster than pure ViT because attention operates on ~512 tokens
    instead of ~2000+ patches.
    """

    def __init__(self, num_classes=2, embed_dim=128, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # CNN stem to reduce spatial dimensions
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # After stem: input (1, 257, 511) → (128, 33, 64) → ~2112 tokens

        # Further pool to reduce token count to manageable size
        self.token_pool = nn.AdaptiveAvgPool2d((8, 16))  # → 128 tokens

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional embedding (128 tokens + 1 CLS)
        self.pos_embed = nn.Parameter(torch.randn(1, 8 * 16 + 1, embed_dim) * 0.02)

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, dropout=dropout)
              for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def _encode(self, x):
        """Shared forward through stem + transformer, returns CLS embedding."""
        B = x.shape[0]

        # CNN feature extraction
        x = self.stem(x)               # [B, embed_dim, H', W']
        x = self.token_pool(x)         # [B, embed_dim, 8, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 128, embed_dim]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 129, embed_dim]
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token

    def forward(self, x):
        cls_out = self._encode(x)
        return self.head(cls_out)

    def get_embedding(self, x):
        """Return CLS token embedding before classification head."""
        return self._encode(x)
