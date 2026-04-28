import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


class Rotary(torch.nn.Module):
    """
    Precompute and cache rotary positional embedding factors (cos/sin) for a given head dimension.

    Parameters
    ----------
    dim : int
        Head dimension (must be even since it is split in half for rotation).
    base : int, optional
        Base used to compute inverse frequencies, by default 10000.
    scaling_factor : float, optional
        Factor to scale context length (e.g., 2.0 for 2048→4096). Uses NTK-aware scaling.

    Attributes
    ----------
    inv_freq : torch.Tensor
        Inverse frequencies of shape ``(dim/2,)``.
    seq_len_cached : int or None
        Sequence length for which cos/sin were last computed.
    cos_cached : torch.Tensor or None
        Cached cosine values with shape ``(seq_len, dim/2)`` in bfloat16.
    sin_cached : torch.Tensor or None
        Cached sine values with shape ``(seq_len, dim/2)`` in bfloat16.
    """

    def __init__(self, dim, base=10000, scaling_factor=1.0):
        super().__init__()
        # NTK-aware RoPE scaling: increase base for longer contexts
        scaled_base = base * scaling_factor
        self.inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        """
        Return cos/sin rotation tensors for input ``x``, computing and caching if needed.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, H, D)`` or similar; only ``T`` is used.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(cos, sin)`` each of shape ``(1, T, 1, D/2)`` and dtype bfloat16.
        """
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary positional embedding rotation to query/key tensors.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(B, T, H, D)`` where ``D`` is even.
    cos : torch.Tensor
        Cosine factors broadcastable to ``x[..., :D/2]``.
    sin : torch.Tensor
        Sine factors broadcastable to ``x[..., :D/2]``.

    Returns
    -------
    torch.Tensor
        Tensor of same shape and dtype as ``x`` with rotary embedding applied.

    Raises
    ------
    AssertionError
        If ``x`` does not have 4 dimensions (expected in multi-head attention).
    """
    assert x.ndim == 4  # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention block using rotary embeddings and RMSNorm on Q/K.

    Parameters
    ----------
    config : object
        Configuration with attributes ``n_head`` and ``n_embd``.

    Attributes
    ----------
    n_head : int
        Number of attention heads.
    n_embd : int
        Embedding dimension.
    head_dim : int
        Dimension per head (``n_embd // n_head``).
    c_q, c_k, c_v : nn.Linear
        Linear projections for queries/keys/values.
    c_proj : nn.Linear
        Output projection (zero-initialized weights).
    rotary : Rotary
        Rotary embedding helper for head_dim.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        """
        Compute causal self-attention over input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, T, C)`` after attention and projection.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))  # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    Feed-forward network block used inside the Transformer.

    Parameters
    ----------
    config : object
        Configuration with attribute ``n_embd``.

    Attributes
    ----------
    c_fc : nn.Linear
        First linear layer expanding to ``4 * n_embd``.
    c_proj : nn.Linear
        Projection back to ``n_embd`` (zero-initialized weights).
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        """
        Apply the two-layer MLP with squared ReLU activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, T, C)``.
        """
        x = self.c_fc(x)
        x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Transformer block composed of (RMSNorm →) Attention + residual and (RMSNorm →) MLP + residual.

    Parameters
    ----------
    config : object
        Configuration passed to submodules.

    Attributes
    ----------
    attn : CausalSelfAttention
        Self-attention sublayer.
    mlp : MLP
        Feed-forward sublayer.
    """

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Apply attention and MLP with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, T, C)``.
        """
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model

class GPT(nn.Module):
    """
    Minimal GPT-like Transformer language model with weight tying.

    Parameters
    ----------
    config : object
        Holds model hyperparameters, expected attributes:
        ``vocab_size``, ``n_embd``, ``n_layer``.

    Attributes
    ----------
    config : object
        Stored configuration.
    transformer : nn.ModuleDict
        Contains token embedding ``wte`` and list of blocks ``h``.
    lm_head : nn.Linear
        Final linear layer tied to token embedding weights.
    gradient_checkpointing : bool
        Whether to use gradient checkpointing to save memory.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False  # Disabled by default

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 

    def forward(self, idx, targets=None, return_logits=True, full_sequence = False):
        """
        Forward pass through the model, optionally computing cross-entropy loss.

        Parameters
        ----------
        idx : torch.Tensor
            Token indices of shape ``(B, T)``.
        targets : torch.Tensor or None, optional
            Target token indices of shape ``(B, T)`` for loss computation.
            If ``None``, no loss is returned, by default None.
        return_logits : bool, optional
            Whether to return logits. If ``False``, returns ``None`` for logits
            to save memory/bandwidth, by default True.

        Returns
        -------
        tuple[torch.Tensor or None, torch.Tensor or None]
            ``(logits, loss)`` where:
            - ``logits`` is float32 of shape ``(B, T, vocab_size)`` when training,
              or ``(B, 1, vocab_size)`` during inference-time optimization, else ``None``.
            - ``loss`` is a scalar tensor when ``targets`` are provided, else ``None``.
        """
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = F.rms_norm(x, (x.size(-1),))
        
        if full_sequence:
            logits = self.lm_head(x)
            logits = logits.float()
            loss   = None

        elif targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
    
    def get_input_embeddings(self):
        return self.transformer.wte

    def forward_from_embeds(self, inputs_embeds, labels=None, return_logits=False):
        x = inputs_embeds
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x) if (return_logits or labels is not None) else None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        return logits, loss
