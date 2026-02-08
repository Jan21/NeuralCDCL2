"""
Decoder-only transformer for CDCL trace prediction.

Pythia-style architecture matching the old working LitGPT config:
- Parallel residual connections (attn + mlp on same input)
- Rotary Position Embeddings (RoPE, 25% of head_dim)
- No attention dropout (dropout_p=0.0)
- No weight tying
- GptNeox-style MLP with GELU
- Uses flash_attn for O(N) memory causal attention on AMD MI250X
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func


def build_rope_cache(
    seq_len: int, n_elem: int, device: torch.device | None = None, base: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cosine/sine caches for RoPE. Returns (cos, sin) each of shape (seq_len, n_elem)."""
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    seq_idx = torch.arange(seq_len, device=device).float()
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    if idx_theta.shape[-1] > n_elem:
        idx_theta = idx_theta[..., :n_elem]
    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE in (B, T, nh, rope_n_elem) layout — no transposes needed.

    Args:
        x: (B, T, nh, rope_n_elem)
        cos: (1, T, 1, rope_n_elem)
        sin: (1, T, 1, rope_n_elem)
    """
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    """Memory-efficient self-attention with RoPE and flash_attn.

    Works entirely in (B, T, nh, hs) layout to avoid transpose copies.
    Uses flash_attn_func (separate q,k,v) to avoid stack allocation.
    """

    def __init__(self, d_model: int, n_heads: int, rotary_percentage: float = 0.25):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_n_elem = int(rotary_percentage * self.head_dim)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Single projection, reshape directly to (B, T, 3, nh, hs)
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.head_dim)
        # unbind returns contiguous copies, allowing qkv to be freed
        q, k, v = qkv.unbind(dim=2)  # each (B, T, nh, hs), contiguous

        # Apply RoPE only to the rotary portion of Q and K
        rn = self.rope_n_elem
        q_roped = apply_rope(q[..., :rn], cos, sin)
        k_roped = apply_rope(k[..., :rn], cos, sin)
        q = torch.cat((q_roped, q[..., rn:]), dim=-1)
        k = torch.cat((k_roped, k[..., rn:]), dim=-1)

        # flash_attn_func takes (B, T, nh, hs) directly — no stack needed
        out = flash_attn_func(q, k, v, causal=True, dropout_p=0.0)
        out = out.reshape(B, T, C)
        return self.out_proj(out)


class GptNeoxMLP(nn.Module):
    """Standard 2-layer MLP with GELU (matching Pythia)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff)
        self.proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class TransformerBlock(nn.Module):
    """
    Parallel residual transformer block (Pythia-style).

    Both attention and MLP receive the same layer-normed input,
    and their outputs are summed into the residual stream.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, rotary_percentage: float = 0.25):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rotary_percentage=rotary_percentage)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = GptNeoxMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Parallel residual: attn and mlp both operate on (different) normed inputs
        x_normed_attn = self.ln1(x)
        attn_out = self.attn(x_normed_attn, cos, sin)

        x_normed_mlp = self.ln2(x)
        mlp_out = self.mlp(x_normed_mlp)

        return x + attn_out + mlp_out


class CDCLTransformer(nn.Module):
    """
    Decoder-only autoregressive transformer for CDCL trace prediction.

    Architecture matches the old working Pythia/LitGPT config:
    - Parallel residual, RoPE (25%), no attention dropout, no weight tying.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 4096,
        rotary_percentage: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, rotary_percentage=rotary_percentage)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # No weight tying (matching Pythia)

        # Build RoPE cache
        head_dim = d_model // n_heads
        rope_n_elem = int(rotary_percentage * head_dim)
        cos, sin = build_rope_cache(max_seq_len, rope_n_elem)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights — scaled init for residual output projections (GPT-NeoX/Pythia)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Scale residual output projections by 1/sqrt(2*n_layers) to prevent
        # residual stream explosion in parallel residual blocks
        residual_std = 0.02 / math.sqrt(2 * len(self.layers))
        for layer in self.layers:
            torch.nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=residual_std)
            torch.nn.init.normal_(layer.mlp.proj.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: unused (kept for interface compatibility)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        # RoPE cos/sin: (1, T, 1, rope_n_elem) for broadcasting with (B, T, nh, rope_n_elem)
        cos = self.cos[:T].unsqueeze(0).unsqueeze(2)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(2)

        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
