"""
conditioning.py
---------------
Interchangeable conditioning modules for ISLES26.

Track A — FiLM (Feature-wise Linear Modulation)
    Takes a hand-crafted metadata vector (dim=4) and produces (gamma, beta)
    scale-shift pairs that modulate decoder bottleneck feature maps.

Track C — LLM Language-grounded Conditioning
    Passes a natural language metadata string through a frozen small LLM
    (Qwen2.5-1.5B or Phi-3-mini) and projects the final hidden state into
    (gamma, beta) pairs with the same interface as Track A.

Both modules expose an identical forward signature:
    forward(meta_vec, meta_text, feature_dim) -> (gamma, beta)
        gamma, beta: (B, feature_dim) tensors for affine modulation

Switching tracks requires only changing cfg.conditioning.track — the model
does not need to know which conditioning module is active.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ── Base interface ─────────────────────────────────────────────────────────────

class BaseConditioner(nn.Module):
    """
    Abstract base for conditioning modules.
    Subclasses implement _encode() to produce a conditioning embedding,
    which is then projected to (gamma, beta) by the shared projection head.
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        # Projection head: shared by both tracks
        # Built lazily on first forward (feature_dim unknown at init time)
        self._proj: Optional[nn.Module] = None

    def _build_projection(self, feature_dim: int) -> None:
        """Build the (gamma, beta) projection head once feature_dim is known."""
        self._proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, feature_dim * 2),  # *2 for gamma + beta
        )
        # Initialise: gamma→1, beta→0 (identity at start of training)
        nn.init.zeros_(self._proj[-1].weight)
        nn.init.zeros_(self._proj[-1].bias)
        self._proj[-1].bias.data[:feature_dim] = 1.0   # gamma bias → 1

        self._proj = self._proj.to(next(self.parameters()).device
                                   if len(list(self.parameters())) > 0
                                   else "cpu")

    def _encode(self, meta_vec: torch.Tensor, meta_text: list[str]) -> torch.Tensor:
        """Return conditioning embedding of shape (B, embed_dim)."""
        raise NotImplementedError

    def forward(
        self,
        meta_vec:    torch.Tensor,   # (B, 4)
        meta_text:   list[str],      # list of B strings
        feature_dim: int,            # C of the bottleneck feature map
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gamma: (B, feature_dim) — multiplicative scale
            beta:  (B, feature_dim) — additive shift
        """
        if self._proj is None or self._proj[-1].out_features != feature_dim * 2:
            self._build_projection(feature_dim)
            self._proj = self._proj.to(meta_vec.device)

        embedding = self._encode(meta_vec, meta_text)         # (B, embed_dim)
        gb        = self._proj(embedding)                     # (B, feature_dim*2)
        gamma, beta = gb.chunk(2, dim=-1)                     # each (B, feature_dim)
        return gamma, beta


# ── Track A: FiLM conditioner ─────────────────────────────────────────────────

class FiLMConditioner(BaseConditioner):
    """
    Track A — FiLM metadata gate.

    Encodes the 4-dim metadata vector [days_norm, is_acute, is_subacute, is_chronic]
    through a small MLP to produce the conditioning embedding.
    """

    def __init__(self, cfg_cond: DictConfig) -> None:
        film_cfg   = cfg_cond.film
        meta_dim   = film_cfg.metadata_dim    # 4
        hidden_dim = film_cfg.hidden_dim       # 64
        super().__init__(embed_dim=hidden_dim, hidden_dim=hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        log.info(f"FiLMConditioner | meta_dim={meta_dim} hidden_dim={hidden_dim}")

    def _encode(self, meta_vec: torch.Tensor, meta_text: list[str]) -> torch.Tensor:
        return self.encoder(meta_vec)   # (B, hidden_dim)


# ── Track C: LLM conditioner ──────────────────────────────────────────────────

class LLMConditioner(BaseConditioner):
    """
    Track C — Language-grounded feature conditioning.

    Uses sentence-transformers (all-MiniLM-L6-v2) for lightweight embedding.
    Still passes natural language metadata strings but uses a 22MB model
    instead of Qwen2.5-1.5B, fitting comfortably in <1GB VRAM.

    The sentence-transformer weights are frozen; only the projection head is trained.
    """

    def __init__(self, cfg_cond: DictConfig) -> None:
        llm_cfg    = cfg_cond.llm
        model_name = llm_cfg.model_name
        embed_dim  = llm_cfg.embedding_dim   # SentenceTransformer output dim
        hidden_dim = llm_cfg.hidden_dim

        super().__init__(embed_dim=embed_dim, hidden_dim=hidden_dim)

        self.model_name  = model_name
        self.freeze_llm  = llm_cfg.freeze_llm
        self._llm        = None    # lazy load on first forward

        log.info(
            f"LLMConditioner | model={model_name} "
            f"embed_dim={embed_dim} freeze={self.freeze_llm}"
        )

    def _load_llm(self, device: torch.device) -> None:
        """Lazy-load sentence-transformer to avoid OOM at import time."""
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading sentence-transformer: {self.model_name} ...")
        self._llm = SentenceTransformer(self.model_name).to(device)

        if self.freeze_llm:
            for param in self._llm.parameters():
                param.requires_grad = False
            log.info("LLM/sentence-transformer weights frozen.")

    def _encode(self, meta_vec: torch.Tensor, meta_text: list[str]) -> torch.Tensor:
        device = meta_vec.device

        if self._llm is None:
            self._load_llm(device)

        # Encode text to embeddings
        with torch.no_grad() if self.freeze_llm else torch.enable_grad():
            embed = self._llm.encode(
                meta_text,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )

        return embed.float()   # cast back to float32 for projection


# ── Factory ───────────────────────────────────────────────────────────────────

def build_conditioner(cfg: DictConfig) -> BaseConditioner:
    """
    Instantiate the correct conditioner from config.
    cfg.conditioning.track: 'A' → FiLM, 'C' → LLM
    """
    track = cfg.conditioning.track.upper()
    if track == "A":
        return FiLMConditioner(cfg.conditioning)
    elif track == "C":
        return LLMConditioner(cfg.conditioning)
    else:
        raise ValueError(
            f"Unknown conditioning track: '{track}'. Must be 'A' or 'C'."
        )