from dataclasses import dataclass

@dataclass
class GPT2_1B:
    """
    Configuration preset for a GPT-2–style Transformer.

    Parameters
    ----------
    vocab_size : int, optional
        Size of the tokenizer vocabulary. Default is 50304.
    n_layer : int, optional
        Number of Transformer blocks. Default is 12.
    n_head : int, optional
        Number of attention heads. Default is 6.  # head dim 128 suggested by @Grad62304977
    n_embd : int, optional
        Embedding (model) dimension. Default is 768.
    """
    vocab_size : int = 50304
    n_layer : int = 52
    n_head : int = 12  # head dim 128 suggested by @Grad62304977
    n_embd : int = 1536
    

@dataclass
class GPT2_4B:
    """
    A100 SXM4 (80GB) batch_size=2 ~11000tokens/GPU single node
    GH200 (92GB): batch_size=4 ~9500tokens/GPU multinode (40)
    H200 (144GB): batch_size=6 ~35000tokens/GPU single node
    H200 (144GB): batch_size=8 ~39000tokens/GPU single node
    
    # NOTE DID I MISCONFIGURE THIS ONE
    B200 (180GB): batch_size=6 ~35000tokens/GPU single node 

    GPT-2–style ~4B config (compatible with your GPT class).
    Head dim stays 128 via n_embd // n_head.
    """
    vocab_size : int = 50304
    n_layer    : int = 20
    n_head     : int = 32     # head_dim = 4096 // 32 = 128
    n_embd     : int = 4096


@dataclass
class GPT2_7B:
    """
    GPT-2–style ~7B config.
    head_dim = 6144 // 48 = 128.
    """
    vocab_size : int = 50304
    n_layer    : int = 32
    n_head     : int = 32     # head_dim = 6144 // 48 = 128
    n_embd     : int = 4096


@dataclass
class GPT2_Tiny:
    """
    Tiny GPT-2–style config for dry-runs and CI (~500K params).
    head_dim = 256 // 2 = 128.
    """
    vocab_size : int = 50304
    n_layer    : int = 4
    n_head     : int = 2      # head_dim = 256 // 2 = 128
    n_embd     : int = 256
