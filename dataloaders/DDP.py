import glob
import os
import torch
import errno

import numpy as np
import pandas as pd

from typing import List, Dict, Any, Optional

MAGIC = 20240520
VERSION = 1

def _peek_data_shard(filename):
    """
    Read only the fixed-size header of a binary data shard and return the
    claimed number of tokens. Supports 256×int32 or 256×int64 headers,
    little- or big-endian. Payload is uint16 tokens.
    """
    total = os.path.getsize(filename)
    with open(filename, "rb") as f:
        head = f.read(24)  # enough for 3×int64

    if len(head) < 12:
        print("ERROR: header too small / truncated .bin file!")
        exit(1)

    candidates = [("<i4", 4), ("<i8", 8), (">i4", 4), (">i8", 8)]
    saw_magic = False
    saw_magic_wrong_version = False

    # First pass: exact match (magic+version) and header ntok consistent with file size
    for fmt, width in candidates:
        need = 12 if width == 4 else 24
        if len(head) < need:
            continue
        dt = np.dtype(fmt)
        magic, version, ntok = np.frombuffer(head[:need], dtype=dt, count=3)
        magic, version, ntok = int(magic), int(version), int(ntok)
        if magic == MAGIC:
            saw_magic = True
            if version != VERSION:
                saw_magic_wrong_version = True
                continue
            header_len = 256 * width
            token_bytes = total - header_len
            if token_bytes >= 0 and token_bytes % 2 == 0:
                computed = token_bytes // 2
                if ntok == computed:
                    return ntok

    # Second pass: accept magic+version, compute ntok from file size if header ntok is off
    for fmt, width in candidates:
        need = 12 if width == 4 else 24
        if len(head) < need:
            continue
        dt = np.dtype(fmt)
        magic, version, _ = np.frombuffer(head[:need], dtype=dt, count=3)
        magic, version = int(magic), int(version)
        if magic == MAGIC and version == VERSION:
            header_len = 256 * width
            token_bytes = total - header_len
            if token_bytes >= 0 and token_bytes % 2 == 0:
                return token_bytes // 2

    # Error handling consistent with your original behavior
    if saw_magic_wrong_version:
        raise AssertionError("unsupported version")
    if not saw_magic:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)

    # If we saw magic+version but couldn't reconcile sizes
    raise ValueError("Could not determine header width/endianness or file is corrupted/truncated.")

def _load_data_shard(filename):
    """
    Load a `.bin` shard that may have either a 256×int32 or 256×int64 header.
    The payload is uint16 tokens.
    """
    with open(filename, "rb") as f:
        # total file size
        f.seek(0, os.SEEK_END)
        total = f.tell()
        f.seek(0)

        # read up to the first 24 bytes (enough for 3 int64s)
        first24 = f.read(24)
        f.seek(0)

        # Try (endianness, width) candidates and pick the one that matches
        # magic, version, and file-size consistency.
        candidates = [("<", 4), ("<", 8), (">", 4), (">", 8)]
        chosen = None

        # strict pass: ntok must match file size
        for endian, width in candidates:
            need = 12 if width == 4 else 24
            if len(first24) < need:
                continue
            dt = np.dtype(endian + ("i4" if width == 4 else "i8"))
            magic, version, ntok = np.frombuffer(first24[:need], dtype=dt, count=3)
            header_len = 256 * width
            token_bytes = total - header_len
            if (int(magic) == MAGIC and int(version) == VERSION
                and token_bytes >= 0 and token_bytes % 2 == 0
                and int(ntok) == token_bytes // 2):
                chosen = (header_len, int(ntok))
                break

        # fallback: accept header but compute ntok from file size (in case header's ntok is off)
        if chosen is None:
            for endian, width in candidates:
                need = 12 if width == 4 else 24
                if len(first24) < need:
                    continue
                dt = np.dtype(endian + ("i4" if width == 4 else "i8"))
                magic, version, _ = np.frombuffer(first24[:need], dtype=dt, count=3)
                header_len = 256 * width
                token_bytes = total - header_len
                if (int(magic) == MAGIC and int(version) == VERSION
                    and token_bytes >= 0 and token_bytes % 2 == 0):
                    chosen = (header_len, token_bytes // 2)
                    break

        assert chosen is not None, (
            "Unrecognized shard header: bad magic/version or unknown header width/endianness."
        )

        header_len, ntok = chosen

        # read payload
        f.seek(header_len)
        tokens = np.fromfile(f, dtype=np.uint16, count=ntok)

    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


def save_checkpoint(model, dataloader, optimizer, scheduler, save_dir="checkpoints"):
    """
    Save a model checkpoint for the current data shard, using a lock file to
    avoid concurrent writes. Optionally run a post-save callback (e.g., HellaSwag).

    Parameters
    ----------
    model : torch.nn.Module
        The model to be checkpointed.
    dataloader : DistributedDataLoader
        Loader whose current shard name is used for checkpoint naming.
    optimizer : list[torch.optim.Optimizer] or torch.optim.Optimizer
        Optimizer(s) whose state dict(s) are saved alongside the model.
    save_dir : str
        Directory to save checkpoints. Default is "checkpoints".

    Notes
    -----
    - Only the first process that successfully creates the lock file writes
      the checkpoint. Others skip silently.
    """
    shard_fname = os.path.basename(dataloader.files[dataloader.current_shard])
    # strip the ".bin" so our checkpoint is named "CC-MAIN-2013-20.pt"
    shard_name = os.path.splitext(shard_fname)[0]

    # prepare checkpoint paths
    ckpt_path = os.path.join(save_dir, shard_name, f"GPT.pt")
    lock_path = ckpt_path + ".lock"

    os.makedirs(os.path.join(save_dir, shard_name), exist_ok=True)

    # --- only the first process to create the lock does the save ---
    try:
        # atomic create, fail if exists
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        # this is the winner — write the checkpoint
        torch.save({
            "model":     model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizer],
            "schedulers" : [sch.state_dict() for sch in scheduler]
        }, ckpt_path)

        print(f"[INFO]: Saved checkpoint for shard {shard_name} → {ckpt_path}")

    except OSError as e:
        if e.errno == errno.EEXIST:
            print("Lock already reached, skipping.")
            # lock already created by another process → skip
            pass
        else:
            raise


class DistributedDataLoader:
    """
    Simple distributed-aware loader for tokenized shards on disk.

    Parameters
    ----------
    filename_pattern : str
        Glob pattern for locating shard files (e.g., ``"data/*.bin"``).
    B : int
        Batch size per process.
    T : int
        Sequence length (tokens per sample).
    process_rank : int
        Index of this process in ``[0, num_processes)``.
    num_processes : int
        Total number of parallel processes.
    on_advance : callable
        Callback executed when advancing to the next shard. Signature:
        ``on_advance(model, dataloader, optimizer)``.

    Attributes
    ----------
    files : list[str]
        Sorted list of shard file paths.
    ntok_total : int
        Total number of tokens across all shards (as declared in headers).
    current_shard : int
        Index of the currently loaded shard.
    current_position : int
        Read pointer within ``self.tokens`` for this process.
    tokens : numpy.ndarray
        Currently loaded token buffer (uint16).

    Notes
    -----
    Each process skips ahead in the token stream by ``process_rank * B * T``
    to avoid overlapping batches across processes.
    """

    def __init__(self, filename_pattern, B, T, process_rank, num_processes, on_advance, skip_files):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.on_advance = on_advance
        
        # glob files that match the pattern
        self.files = np.array(sorted(glob.glob(filename_pattern)))
        if skip_files is not None:
            self.files = self.files[self.files > skip_files]
            
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            # ensure each shard has enough tokens for all processes to draw at least one batch
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        """
        Reset loader state to the beginning of the first shard for this process.

        Notes
        -----
        Sets ``current_shard`` to 0 and ``current_position`` to the process-specific
        offset, then loads the first shard's tokens.
        """
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def current_file_month(self) -> Optional[str]:
        """
        Extract the month identifier from the current shard filename.
        
        Returns
        -------
        str or None
            Month string (e.g., '2013-05') if filename matches YYYY-MM pattern,
            otherwise None.
        """
        if self.current_shard >= len(self.files):
            return None
        fname = os.path.basename(self.files[self.current_shard])
        # Expected format: YYYY-MM.bin or YYYY-MM_partN.bin
        name = os.path.splitext(fname)[0]
        # Handle split files: 2013-12_part1 -> 2013-12
        if '_part' in name:
            name = name.split('_part')[0]
        # Basic validation: should be 7 chars like "2013-05"
        if len(name) == 7 and name[4] == '-':
            return name
        return None

    def advance(self, model, optimizer, scheduler):
        """
        Move to the next shard and invoke the advance callback.

        Parameters
        ----------
        model : torch.nn.Module or None
            Model passed through to ``on_advance``. If ``None``, the callback
            is skipped.
        optimizer : list[torch.optim.Optimizer] or torch.optim.Optimizer
            Same object(s) passed through to ``on_advance``.
        """
        if model is not None:
            self.on_advance(model, self, optimizer, scheduler)

        # advance to next data shard (wrap-around modulo file count)
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, model, optimizer, scheduler):
        """
        Retrieve the next (inputs, targets) batch pair and advance internal
        position, potentially loading the next shard.

        Parameters
        ----------
        model : torch.nn.Module or None
            Forwarded to :meth:`advance` when a shard boundary is crossed.
        optimizer : list[torch.optim.Optimizer] or torch.optim.Optimizer
            Forwarded to :meth:`advance`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(x, y)`` tensors of shape ``(B, T)`` on CUDA. ``x`` are inputs,
            ``y`` are next-token targets.

        Notes
        -----
        - Uses an int32 intermediate NumPy cast to ensure compatibility with
          PyTorch's ``long`` dtype.
        - When the end of the shard is reached for *all* processes, ``advance``
          is called to load the next shard and optionally save a checkpoint.
        """
        B = self.B
        T = self.T
        # slice B*T+1 tokens: the last one is used only as the first target
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance(model, optimizer, scheduler)
        return x.cuda(), y.cuda()
    
class DistributedDataLoaderSP:
    
    def __init__(self,
                 filename_pattern,
                 B,
                 T,
                 process_rank,
                 num_processes,
                 model_time,
                 sp_len,
                 pad_token_id: int = 50256,
                 truncate_side: str = "right",
                 is_training : bool = True,
                 decoder : bool = False):
        
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.sp_len = sp_len
        self.pad_token_id = pad_token_id
        self.truncate_side = truncate_side
        self.decoder = decoder
        
        assert 0 <= self.sp_len < self.T, "soft_prompt_len must be >=0 and < T"
        self.max_text_len = self.T - self.sp_len

        # glob files that match the pattern
        self.files = np.array(sorted(glob.glob(filename_pattern)))
        
        self.files = self.files[self.files < '/'.join(filename_pattern.split('/')[:-1])+ f"/{model_time}.parquet"]
        
        if is_training:
            self.files = self.files[:-1]
        else:
            self.files = [self.files[-1]]
        
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
        
        tot_ex = 0
        max_tok = 0
        for file in self.files:
            n_ex, max_tok_file = self.peak_shard(file)
            tot_ex += n_ex
            if max_tok_file > max_tok:
                max_tok = max_tok_file
                
        self.max_tok = max_tok
            
        print(f"We have {tot_ex} examples at disposition")
        print(f"Max token size {max_tok}")
            
        self.current_shard = 0
        self.current_position = 0 
        self.epoch = 0
        
        # buffers for active shard
        self._tokens: List[List[int]] = []
        self._C: Optional[np.ndarray] = None
        self._dates: List[str] = []
        self._N: int = 0
        self._order: Optional[np.ndarray] = None

        # kick things off
        self.reset()
        
    def load_shard(self, filename: str):
        """
        Load a Parquet shard and prepare internal buffers.
        Expects columns: 'date' (YYYYMMDD), 'tokens' (list[int]), 'C' (list[float])
        """
        df = pd.read_parquet(filename)
        if not {"date", "tokens", "C"}.issubset(df.columns):
            raise ValueError(f"{filename} missing required columns (have {df.columns.tolist()})")

        # ensure clean types
        dates = df["date"].astype(str).tolist()
        tokens_col = df["tokens"].tolist()
        C_col = df["C"].tolist()

        # convert C to 2D float32 array
        try:
            C_mat = np.array([np.array(c, dtype=np.float32) for c in C_col], dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Column 'C' must be list-like per row. Got error: {e}")

        # ensure tokens are list[int]
        toks_list: List[List[int]] = []
        for i, t in enumerate(tokens_col):
            if isinstance(t, (list, tuple, np.ndarray)):
                toks = [int(x) for x in t]
            else:
                raise ValueError(f"Row {i} 'tokens' must be list-like, got {type(t)}")
            toks_list.append(toks)

        # store
        self._tokens = toks_list
        self._C = C_mat
        self._dates = dates
        self._N = len(self._tokens)
        self._order = np.arange(self._N)
        
    def peak_shard(self, filename : str):
        
        df = pd.read_parquet(filename)
        
        max_tok_file = df["tokens"].apply(len).max()
        
        return df.shape[0], max_tok_file
            
    def _advance_shard(self):
        """Load next shard; if we pass the end, wrap and increment epoch."""
        self.current_shard += 1
        if self.current_shard >= len(self.files):
            self.current_shard = 0
            self.epoch += 1
        self.load_shard(self.files[self.current_shard])
        # reset position for this shard to the rank's offset
        self.current_position = self.process_rank * self.B
        
    def reset(self):
        """
        Reset loader state to the beginning of the first shard for this process.

        Notes
        -----
        Sets ``current_shard`` to 0 and ``current_position`` to the process-specific
        offset, then loads the first shard's tokens.
        """
        self.current_shard = 0
        self.current_position = self.process_rank * self.B
        self.epoch = 0
        self.load_shard(self.files[self.current_shard])
        
    def _prepare_batch_indices(self) -> Optional[np.ndarray]:
        """
        Return indices for the next mini-batch for this rank, or None if shard exhausted.
        Uses rank-based striding with step B*num_processes.
        """
        if self._N == 0:
            return None

        start = self.current_position
        end = start + self.B
        if end > self._N:
            return None  # force caller to advance shard

        idx_in_shard = self._order[start:end]
        # advance pointer for next call
        self.current_position += self.B * self.num_processes
        return idx_in_shard

    def _truncate(self, toks: List[int]) -> List[int]:
        """Truncate a token list to fit max_text_len according to truncate_side."""
        if len(toks) <= self.max_text_len:
            return toks
        if self.truncate_side == "left":
            return toks[-self.max_text_len:]
        else:
            return toks[:self.max_text_len]
        
    def _collate(self, batch_tokens: List[List[int]], batch_C: np.ndarray) -> Dict[str, Any]:
        """Pad/truncate to uniform length and build tensors."""
        # truncate and compute lengths
        trunc = [self._truncate(t) for t in batch_tokens]
        lengths = [len(t) for t in trunc]
        if self.decoder:
            T_text = self.max_tok
        else:
            T_text = max(1, lengths)  # avoid zero
        B = len(trunc)

        input_ids = torch.full((B, T_text), self.pad_token_id, dtype=torch.long)
        input_ids_reverse = torch.full((B, T_text), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T_text), dtype=torch.long)
        labels = torch.full((B, T_text), -100, dtype=torch.long)

        for i, t in enumerate(trunc):
            L = len(t)
            if L == 0:
                continue
            ids = torch.tensor(t, dtype=torch.long)
            input_ids[i, :L] = ids
            input_ids_reverse[i,-L:] = ids
            attention_mask[i, :L] = 1
            labels[i, :L] = ids
            labels[i, 0] = -100  # mask first token target

        c = torch.tensor(batch_C, dtype=torch.float32)
        
        attention_mask = (attention_mask != 0 )
        
        if self.decoder:
            c = input_ids_reverse
        
        return {
            "input_ids": input_ids,                     # [B, T_text]
            "attention_mask": attention_mask,           # [B, T_text]
            "c": c,   
            "labels" : labels# [B, D]
        }
        
    def next_batch(self) -> Dict[str, Any]:
        """
        Return the next mini-batch for this rank.
        If the current shard can't provide a full mini-batch for all ranks, advance shards until it can.
        """
        while True:
            idx = self._prepare_batch_indices()
            if idx is None:
                # shard exhausted for this rank; move on
                self._advance_shard()
                continue

            # gather rows
            btoks = [self._tokens[i] for i in idx]
            bC = self._C[idx, :]
            return self._collate(btoks, bC)
        

if __name__ == "__main__":
    
    loader = DistributedDataLoaderSP(
                filename_pattern="/capstor/scratch/cscs/sjohanne/WSJ_token/*.parquet",
                B=8,
                T=2048,                 # full model context (incl. soft prompt)
                process_rank=1,
                num_processes=1,
                model_time="202210",  # keep files strictly before this yyyyMM or yyyyMMdd name
                sp_len=100,    # set to 0 if none
                pad_token_id=50256,
                decoder = True
            )
    breakpoint()
    
    batch = loader.next_batch()
    
    
    breakpoint()
    
    
    
    tokens = _load_data_shard('/capstor/scratch/cscs/sjohanne/fineweb_monthly_7B/2015-01.bin')
    breakpoint()