"""
model.py
--------
nnU-Net-inspired 3D encoder-decoder with pluggable conditioning.

Architecture:
  - 3D U-Net with residual blocks (nnU-Net full-resolution style)
  - Conditioning module (FiLM or LLM) injected at the bottleneck
  - Deep supervision heads on all decoder scales (nnU-Net default)

Design notes:
  - We implement the backbone directly rather than wrapping nnU-Net's CLI,
    giving full control over the conditioning injection point and loss computation.
  - The architecture mirrors nnU-Net's 3D full-res defaults for the ATLAS
    fingerprint (isotropic 1mm, 128³ patches).
  - Conditioning is applied via FiLM modulation: feature maps are scaled and
    shifted by (gamma, beta) from the conditioner before the final decoder block.

Exported:
  ISLES26Model(cfg) — the full model
  build_model(cfg)  — factory with weight init logging
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .conditioning import build_conditioner, BaseConditioner

log = logging.getLogger(__name__)


# ── Building blocks ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block: Conv → InstanceNorm → LeakyReLU → Conv → InstanceNorm,
    with a 1×1×1 skip projection if in_ch != out_ch.
    InstanceNorm is used (not BatchNorm) for small-batch medical imaging.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(0.01, inplace=True)

        self.skip  = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm2(self.conv2(self.act(self.norm1(self.conv1(x))))) + self.skip(x))


class EncoderBlock(nn.Module):
    """Encoder stage: ResBlock + strided conv for downsampling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res     = ResBlock(in_ch, out_ch)
        self.down    = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.down_norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act     = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res(x)
        out  = self.act(self.down_norm(self.down(skip)))
        return out, skip


class DecoderBlock(nn.Module):
    """
    Decoder stage: trilinear upsample → concat skip → ResBlock.
    Returns both the output and an intermediate feature map for deep supervision.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.res = ResBlock(in_ch + skip_ch, out_ch)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd spatial dims during encoding
        # after upsample, x might not exactly match skip spatially
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        # Debug: log shapes before cat
        import sys
        print(f"[DecoderBlock] x.shape={x.shape}, skip.shape={skip.shape}", file=sys.stderr)

        return self.res(torch.cat([x, skip], dim=1))


class SegmentationHead(nn.Module):
    """1×1×1 conv to produce logits for deep supervision."""

    def __init__(self, in_ch: int, n_classes: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── FiLM application ──────────────────────────────────────────────────────────

def apply_film(
    x:     torch.Tensor,   # (B, C, H, W, D)
    gamma: torch.Tensor,   # (B, C)
    beta:  torch.Tensor,   # (B, C)
) -> torch.Tensor:
    """
    Apply FiLM modulation: x = gamma * x + beta
    gamma and beta are broadcast over spatial dims.
    """
    g = gamma.view(*gamma.shape, 1, 1, 1)
    b = beta.view(*beta.shape,   1, 1, 1)
    return g * x + b


# Model size configurations (parameter count estimates)
MODEL_CONFIGS = {
    "tiny":  {"enc_ch": [1, 16, 32,  64,  128], "bottleneck": 160},  # ~3M params
    "small": {"enc_ch": [1, 16, 32,  64,  128], "bottleneck": 256},  # ~6M params
    "base":  {"enc_ch": [1, 32, 64, 128,  256], "bottleneck": 320},  # ~19M params
}


# ── Full model ────────────────────────────────────────────────────────────────

class ISLES26Model(nn.Module):
    """
    3D U-Net with conditioning injection at the bottleneck.

    Deep supervision: segmentation heads at decoder scales 0-3.
    Conditioning: FiLM or LLM gate applied to bottleneck features.

    Model size is configurable via cfg.model.size: "tiny" | "small" | "base"
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # Get model size config (default to "base" if not specified)
        mc = MODEL_CONFIGS.get(cfg.model.get("size", "base"), MODEL_CONFIGS["base"])
        self.ENCODER_CHANNELS = mc["enc_ch"]
        self.BOTTLENECK_CH    = mc["bottleneck"]

        ch = self.ENCODER_CHANNELS
        bn = self.BOTTLENECK_CH

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc0 = EncoderBlock(ch[0], ch[1])   # 1   → 32
        self.enc1 = EncoderBlock(ch[1], ch[2])   # 32  → 64
        self.enc2 = EncoderBlock(ch[2], ch[3])   # 64  → 128
        self.enc3 = EncoderBlock(ch[3], ch[4])   # 128 → 256

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ResBlock(ch[4], bn)    # 256 → 320

        # ── Post-FiLM normalisation to prevent feature scale explosion ──────────
        self.post_film_norm = nn.InstanceNorm3d(bn, affine=True)

        # ── Conditioning ─────────────────────────────────────────────────────
        self.conditioner: BaseConditioner = build_conditioner(cfg)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec3 = DecoderBlock(bn,    ch[4], ch[4])  # 320+256 → 256
        self.dec2 = DecoderBlock(ch[4], ch[3], ch[3])  # 256+128 → 128
        self.dec1 = DecoderBlock(ch[3], ch[2], ch[2])  # 128+64  → 64
        self.dec0 = DecoderBlock(ch[2], ch[1], ch[1])  # 64+32   → 32

        # ── Deep supervision heads ────────────────────────────────────────────
        # Heads ordered coarse→fine to match DEEP_SUP_WEIGHTS
        self.ds_heads = nn.ModuleList([
            SegmentationHead(ch[1]),   # finest  (dec0 output)
            SegmentationHead(ch[2]),   # dec1
            SegmentationHead(ch[3]),   # dec2
            SegmentationHead(ch[4]),   # coarsest (dec3)
        ])

        self._log_arch()

    def _log_arch(self) -> None:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"ISLES26Model | track={type(self.conditioner).__name__} "
            f"| trainable params={n_params:,}"
        )

    def forward(
        self,
        x:          torch.Tensor,   # (B, 1, H, W, D)
        meta_vec:   torch.Tensor,   # (B, 5)  [days_norm, is_acute, is_subacute, is_chronic, confirmed_chronic]
        meta_text:  list[str],      # list of B strings
        chronicity: Optional[list[str]] = None,  # for hooks
        uid:        Optional[list[str]] = None,  # for hooks
    ) -> list[torch.Tensor]:
        """
        Returns a list of logit tensors for deep supervision,
        ordered finest→coarsest: [full_res, /2, /4, /8]

        At inference, use only outputs[0] (full resolution).
        """
        # ── Encode ────────────────────────────────────────────────────────────
        x, s0 = self.enc0(x)
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)

        # ── Bottleneck ────────────────────────────────────────────────────────
        x_before_film = self.bottleneck(x)              # (B, 320, H/16, W/16, D/16)

        # ── Conditioning injection ────────────────────────────────────────────
        gamma, beta = self.conditioner(meta_vec, meta_text, feature_dim=x_before_film.shape[1])

        # Hook for interpretability (read-only, zero overhead when not attached)
        if getattr(self, "_film_hook", None) is not None:
            self._film_hook(gamma, beta, meta_text, chronicity or [])

        x_film = apply_film(x_before_film, gamma, beta)
        x_after_film = self.post_film_norm(x_film)   # prevent feature scale explosion

        # Hook for bottleneck embeddings (read-only, zero overhead when not attached)
        # x_pre = before FiLM, x_post = after FiLM
        if getattr(self, "_embed_hook", None) is not None:
            self._embed_hook(x_before_film.clone(), x_after_film, chronicity or [], uid or [])

        # ── Decode ────────────────────────────────────────────────────────────
        d3 = self.dec3(x_after_film,  s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        # ── Deep supervision logits ───────────────────────────────────────────
        return [
            self.ds_heads[0](d0),   # finest
            self.ds_heads[1](d1),
            self.ds_heads[2](d2),
            self.ds_heads[3](d3),   # coarsest
        ]

    def predict(
        self,
        x:         torch.Tensor,
        meta_vec:  torch.Tensor,
        meta_text: list[str],
        chronicity: Optional[list[str]] = None,
        uid:       Optional[list[str]] = None,
    ) -> torch.Tensor:
        """
        Inference-only: returns binary mask from finest-scale logits.
        No deep supervision — returns (B, 1, H, W, D) uint8 mask.
        """
        with torch.no_grad():
            logits = self.forward(x, meta_vec, meta_text, chronicity, uid)[0]
            probs  = torch.softmax(logits, dim=1)[:, 1:2]   # class-1 probability
            return (probs > 0.5).to(torch.uint8)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> ISLES26Model:
    """Instantiate and weight-initialise the model."""
    model = ISLES26Model(cfg)

    # Kaiming init for conv layers (except conditioning projection, already init'd)
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm3d) and m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    log.info("Model weight initialisation complete.")
    return model