"""
Tensor-Parallel GPT model.

Architecture overview
---------------------
The residual stream tensor x of shape (B, T, n_embd) is REPLICATED on
every TP rank throughout the entire forward pass.  Only intermediate
tensors *inside* CausalSelfAttentionTP and MLPTP are sharded.

This means all of the following are identical on every rank and need no
special treatment:
  - Token embedding (wte / lm_head)
  - RMSNorm
  - Residual additions
  - Cross-entropy loss

Inside each transformer block the communication pattern is:

  x (replicated)
  │
  ColumnParallelLinear  ← _CopyToAllTPRanks  (identity fwd, all_reduce bwd)
  │  local matmul: no comm
  ↓ (sharded along output/head dim)
  ... local ops (RoPE, SDPA, activation) ...
  ↓ (still sharded)
  RowParallelLinear     ← _ReduceFromAllTPRanks (all_reduce fwd, identity bwd)
  │  local matmul + all_reduce
  ↓
  x (replicated)

Total communication cost per transformer block: one all_reduce forward,
one all_reduce backward.  This matches Megatron-LM's minimal pattern.

Usage
-----
    from models.GPT_TP import GPTTP
    from training.distributed_computing.ParallelContext import build_parallel_context

    parallel = build_parallel_context(tp_size)
    model = GPTTP(config, tp_group=parallel.tp_group, tp_size=parallel.tp_size)
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.GPT import Rotary, apply_rotary_emb
from training.distributed_computing.tp_layers import ColumnParallelLinear, RowParallelLinear


class CausalSelfAttentionTP(nn.Module):
    """
    Multi-head causal self-attention with heads split across TP ranks.

    Each rank owns n_head // tp_size heads.

    head_dim = n_embd // n_head is UNCHANGED — we split heads, not the head
    dimension.  Rotary embeddings therefore require no modification.

    Parameters
    ----------
    config : GPT2Config
        Model config (vocab_size, n_layer, n_head, n_embd).
    tp_group : dist.ProcessGroup
        TP process group for this rank.
    tp_size : int
        Number of tensor-parallel ranks.
    """

    def __init__(self, config, tp_group, tp_size: int):
        super().__init__()

        assert config.n_head % tp_size == 0, (
            f"n_head={config.n_head} must be divisible by tp_size={tp_size}"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head   # unchanged
        self.n_head_local = config.n_head // tp_size     # heads owned by this rank

        # Each projection outputs n_embd/tp features = n_head_local * head_dim.
        # ColumnParallelLinear stores weight (n_embd//tp, n_embd) locally.
        self.c_q = ColumnParallelLinear(config.n_embd, config.n_embd, tp_group, tp_size, bias=False)
        self.c_k = ColumnParallelLinear(config.n_embd, config.n_embd, tp_group, tp_size, bias=False)
        self.c_v = ColumnParallelLinear(config.n_embd, config.n_embd, tp_group, tp_size, bias=False)

        # Output projection: takes sharded input (n_embd//tp) → full output (n_embd).
        # RowParallelLinear stores weight (n_embd, n_embd//tp) locally.
        self.c_proj = RowParallelLinear(config.n_embd, config.n_embd, tp_group, tp_size, bias=False)
        # Zero-init the output projection (same as original GPT)
        self.c_proj.weight.data.zero_()

        self.rotary = Rotary(self.head_dim)  # head_dim is unchanged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Q/K/V projections — output is (B, T, n_embd // tp_size) on this rank
        # Reshape to (B, T, n_head_local, head_dim) for attention
        q = self.c_q(x).view(B, T, self.n_head_local, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head_local, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head_local, self.head_dim)

        # Rotary embeddings — applied to local heads only (head_dim unchanged)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention over local heads
        # Input: (B, n_head_local, T, head_dim) — transpose T and n_head_local
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )

        # Reshape back: (B, T, n_embd // tp_size)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head_local * self.head_dim)

        # RowParallelLinear: local matmul + all_reduce → (B, T, n_embd) replicated
        return self.c_proj(y)


class MLPTP(nn.Module):
    """
    Feed-forward network with the hidden dimension split across TP ranks.

    Each rank owns 4 * n_embd // tp_size hidden units.

    Parameters
    ----------
    config : GPT2Config
        Model config.
    tp_group : dist.ProcessGroup
        TP process group for this rank.
    tp_size : int
        Number of tensor-parallel ranks.
    """

    def __init__(self, config, tp_group, tp_size: int):
        super().__init__()

        # Expand: (n_embd) → (4 * n_embd // tp_size) — sharded
        self.c_fc = ColumnParallelLinear(config.n_embd, 4 * config.n_embd, tp_group, tp_size, bias=False)

        # Project back: (4 * n_embd // tp_size) → (n_embd) — all_reduce
        self.c_proj = RowParallelLinear(4 * config.n_embd, config.n_embd, tp_group, tp_size, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ColumnParallel: (B, T, n_embd) → (B, T, 4*n_embd//tp)
        x = self.c_fc(x)
        # Activation on local shard
        x = F.relu(x).square()
        # RowParallel: (B, T, 4*n_embd//tp) → (B, T, n_embd) via all_reduce
        return self.c_proj(x)


class BlockTP(nn.Module):
    """
    Transformer block using TP-aware attention and MLP.

    The residual stream x stays (B, T, n_embd) and is replicated at all times.
    RMSNorm and residual adds are pure local ops — no communication.
    """

    def __init__(self, config, tp_group, tp_size: int):
        super().__init__()
        self.attn = CausalSelfAttentionTP(config, tp_group, tp_size)
        self.mlp = MLPTP(config, tp_group, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class GPTTP(nn.Module):
    """
    GPT model with Tensor Parallelism.

    Drop-in replacement for GPT (same forward signature) but internally
    uses ColumnParallelLinear / RowParallelLinear in attention and MLP.

    The embedding wte and lm_head remain full-size on every rank (they
    operate on the full residual stream).  Weight tying is preserved.

    Parameters
    ----------
    config : GPT2Config
        Model hyperparameters.
    tp_group : dist.ProcessGroup
        The TP process group this rank belongs to.
    tp_size : int
        Number of tensor-parallel ranks (must divide n_head and 4*n_embd).
    """

    def __init__(self, config, tp_group, tp_size: int):
        super().__init__()
        self.config = config
        self.tp_size = tp_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([BlockTP(config, tp_group, tp_size) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: lm_head shares weights with token embedding
        self.transformer.wte.weight = self.lm_head.weight

    def num_params_total(self) -> int:
        """
        True total parameter count, accounting for TP sharding.

        wte / lm_head are full-size and replicated on every rank — count once
        (weight tying means lm_head doesn't appear separately in named_parameters).

        Every weight inside the transformer blocks is sharded: each rank holds
        1/tp_size of the full matrix, so we multiply the local count by tp_size.
        """
        n_replicated = self.transformer.wte.weight.numel()
        n_sharded_local = sum(
            p.numel() for block in self.transformer.h for p in block.parameters()
        )
        return n_replicated + n_sharded_local * self.tp_size

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)          # (B, T, n_embd) — replicated

        for block in self.transformer.h:
            x = block(x)                       # (B, T, n_embd) — stays replicated

        x = F.rms_norm(x, (x.size(-1),))      # local, no comm

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None

        if not return_logits:
            logits = None

        return logits, loss
