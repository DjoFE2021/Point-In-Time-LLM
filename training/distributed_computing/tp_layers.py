"""
Tensor-Parallel linear layers.

Design
------
A standard linear Y = X W^T can be split across tp_size GPUs in two ways:

  ColumnParallelLinear  — splits the *output* dimension (W rows).
    Each rank owns W[tp_rank * out/tp : (tp_rank+1) * out/tp, :].
    Input X is replicated → no forward comm.
    Backward grad_X is partial → all_reduce needed in backward.

  RowParallelLinear     — splits the *input* dimension (W columns).
    Each rank owns W[:, tp_rank * in/tp : (tp_rank+1) * in/tp].
    Input X is already sharded (comes from ColumnParallelLinear).
    Output is a partial sum → all_reduce in forward.

They always appear as a pair:

    x (replicated)
      │
  ColumnParallelLinear    [no forward comm; all_reduce in backward]
      │
    y (sharded along out_features)
      │
    (local ops: attention, activation, etc.)
      │
  RowParallelLinear       [all_reduce in forward; no backward comm]
      │
    z (replicated)

This gives exactly one all_reduce per sub-layer in the forward pass and
one all_reduce per sub-layer in the backward pass — matching Megatron-LM.

Compatibility
-------------
Both layers inherit from nn.Linear so they can be used as drop-in
replacements. The only difference at construction time is that you also
pass `tp_group` (a dist.ProcessGroup) and `tp_size` (int).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Autograd communication primitives
# ---------------------------------------------------------------------------

class _CopyToAllTPRanks(torch.autograd.Function):
    """
    Forward : identity  (input passes through unchanged)
    Backward: all_reduce gradient across the TP group

    Why here and not somewhere else?
    ---------------------------------
    In a ColumnParallelLinear, every rank starts with the *same* X and
    computes a local grad_X = grad_Y @ W_local^T.  Because W is split
    across ranks, each rank's grad_X is only a *partial* contribution to
    the true gradient.  We need to sum those partials.

    Placing this identity-forward / all_reduce-backward op at the *input*
    of the column-parallel layer achieves exactly that: the all_reduce
    fires automatically during the backward pass without any manual calls
    in the training loop.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group) -> torch.Tensor:
        ctx.tp_group = tp_group
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # Sum partial grad_X contributions from all TP ranks
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad, None  # None for tp_group (not a tensor)


class _ReduceFromAllTPRanks(torch.autograd.Function):
    """
    Forward : all_reduce (sum partial outputs across the TP group)
    Backward: identity   (gradient flows back unchanged)

    Why here and not somewhere else?
    ---------------------------------
    In a RowParallelLinear, each rank computes partial_out = X_local @ W_local^T.
    These partial outputs must be summed to get the correct full output.

    The backward of an all_reduce sum is just to pass the incoming gradient
    through unchanged (each rank contributed to the sum equally, so each
    rank's gradient is the full grad_out — no further comm needed).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group) -> torch.Tensor:
        # Clone before in-place all_reduce so autograd graph stays clean
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None  # identity; None for tp_group


# ---------------------------------------------------------------------------
# Parallel linear layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Linear):
    """
    Linear layer with the output dimension split across tp_size ranks.

    Parameters
    ----------
    in_features : int
        Full input feature dimension (same on every rank).
    out_features : int
        Full output feature dimension (must be divisible by tp_size).
        Each rank holds out_features // tp_size rows of the weight matrix.
    tp_group : dist.ProcessGroup
        The process group containing all ranks in this TP group.
    tp_size : int
        Number of tensor-parallel ranks.
    **kwargs
        Forwarded to nn.Linear (e.g. bias=False).

    Shape
    -----
    Input  : (B, T, in_features)   — replicated on every TP rank
    Output : (B, T, out_features // tp_size)  — each rank holds a shard
    Weight : (out_features // tp_size, in_features)  — local shard
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group,
        tp_size: int,
        **kwargs,
    ):
        if out_features % tp_size != 0:
            raise ValueError(
                f"out_features={out_features} must be divisible by tp_size={tp_size}"
            )
        # nn.Linear stores a weight of shape (out_features_local, in_features)
        super().__init__(in_features, out_features // tp_size, **kwargs)
        self.tp_group = tp_group
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tag x so that all_reduce fires on grad_x during backward
        x = _CopyToAllTPRanks.apply(x, self.tp_group)
        # Local matrix multiply — no communication
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Linear):
    """
    Linear layer with the input dimension split across tp_size ranks.

    Parameters
    ----------
    in_features : int
        Full input feature dimension (must be divisible by tp_size).
        Each rank receives in_features // tp_size input features.
    out_features : int
        Full output feature dimension (same on every rank after all_reduce).
    tp_group : dist.ProcessGroup
        The process group containing all ranks in this TP group.
    tp_size : int
        Number of tensor-parallel ranks.
    **kwargs
        Forwarded to nn.Linear (e.g. bias=False).

    Shape
    -----
    Input  : (B, T, in_features // tp_size)  — each rank holds a shard
    Output : (B, T, out_features)            — replicated on every TP rank
    Weight : (out_features, in_features // tp_size)  — local shard
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group,
        tp_size: int,
        **kwargs,
    ):
        if in_features % tp_size != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by tp_size={tp_size}"
            )
        # nn.Linear stores a weight of shape (out_features, in_features_local)
        super().__init__(in_features // tp_size, out_features, **kwargs)
        self.tp_group = tp_group
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matrix multiply on sharded input
        out = F.linear(x, self.weight, self.bias)
        # Sum partial outputs across all TP ranks → full output replicated everywhere
        return _ReduceFromAllTPRanks.apply(out, self.tp_group)
