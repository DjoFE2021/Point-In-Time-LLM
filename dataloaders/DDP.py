"""Distributed data loader for GPT pre-training: streams monthly .bin token shards and calls a
month-end callback before any rank reads the next month.

The .bin shard format (magic 20240520) and the loader design follow llm.c / modded-nanogpt (MIT licensed;
see THIRD_PARTY_NOTICES.md).
"""
import glob
import os
import torch

import numpy as np

from typing import Optional

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

    # No usable header: report a wrong version or a missing magic number
    if saw_magic_wrong_version:
        raise AssertionError("unsupported version")
    if not saw_magic:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Do --data-dir / --val-dir point to .bin shards (glob patterns such as '<dir>/*.bin')?")
        print("---> HINT: Shards are built with data/get_train_set.py and data/get_validation_set.py (see README)")
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

        # Pick the (endianness, header width) matching magic, version and file size
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
    on_month_end : callable or None
        Called on every rank when a month's data runs out, before any rank
        reads data from the next month (see :meth:`advance`). Signature:
        ``on_month_end(model, dataloader, optimizer, scheduler)``.
    skip_files : str or None
        Resume support: only shards whose path sorts after this one are used.

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
    to avoid overlapping batches across processes. All processes switch shards
    on the same call, so they are always reading the same month.
    """

    def __init__(self, filename_pattern, B, T, process_rank, num_processes, on_month_end, skip_files):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.on_month_end = on_month_end
        
        # glob files that match the pattern
        if os.path.isdir(filename_pattern):
            raise ValueError(f"{filename_pattern} is a directory; pass a glob pattern such as '{filename_pattern}/*.bin'")
        self.files = np.array(sorted(glob.glob(filename_pattern)))
        if skip_files is not None:
            self.files = self.files[self.files > skip_files]
            
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # Validate every shard header and count the total number of tokens
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            # ensure each shard has enough tokens for all processes to draw at least one batch
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

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
        return self._file_month(self.current_shard)

    def _file_month(self, shard_idx: int) -> Optional[str]:
        """Month string of shard ``shard_idx`` (e.g. '2013-05'), or None if the name has no YYYY-MM."""
        fname = os.path.basename(self.files[shard_idx])
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
        Move to the next shard, calling ``on_month_end`` first if this ends a month.

        A month ends when the next shard belongs to a different month or the
        loader wraps around to the first shard. The callback runs before the
        next shard is loaded, so no rank has read the next month's data yet.

        Parameters
        ----------
        model : torch.nn.Module or None
            Model passed through to ``on_month_end``. If ``None``, the callback
            is skipped.
        optimizer : list[torch.optim.Optimizer] or torch.optim.Optimizer
            Same object(s) passed through to ``on_month_end``.
        """
        next_shard = (self.current_shard + 1) % len(self.files)
        month = self.current_file_month()
        month_ends = month is not None and (next_shard == 0 or self._file_month(next_shard) != month)
        if model is not None and self.on_month_end is not None and month_ends:
            self.on_month_end(model, self, optimizer, scheduler)

        # advance to next data shard (wrap-around modulo file count)
        self.current_shard = next_shard
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
        - When the shard has no room for another full round (one batch per
          process), ``advance`` is called on every process in the same call.
        """
        B = self.B
        T = self.T
        # Slice B*T+1 tokens: the extra token is only used as the last target
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        # Check whether the next round fits from the round's start (rank 0's position), not this
        # rank's own position, so that every rank switches shard on the same call. Otherwise
        # higher ranks would move on to the next month before rank 0.
        round_start = self.current_position - self.process_rank * B * T
        if round_start + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance(model, optimizer, scheduler)
        return x.cuda(), y.cuda()
