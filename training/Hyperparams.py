"""Pre-training hyperparameter presets (data globs, batch sizes, LR, eval cadence).

train_gpt.py picks one via --config. The data globs below are placeholders:
pass --data-dir / --val-dir to point at your shards.
"""
from dataclasses import dataclass

@dataclass
class CSCS_60GPU:
    """
    60-GPU training config with 1024 context.
    Tokens/step = 480 * 1024 = 491,520.
    """
    # data
    input_bin     : str = "fineweb_pit/train/*.bin"
    input_val_bin : str = "fineweb_pit/val/fineweb_val_*.bin"

    # optimization
    batch_size        : int = 8*60  # batch size, in sequences, across all devices
    device_batch_size : int = 8  # batch size, in sequences, per device (12 was used on 8 GPUs)
    sequence_length   : int = 1024  # sequence length, in tokens
    learning_rate     : float = 0.0036 / 2
    warmup_iters      : int = 0
    weight_decay      : float = 0

    # eval/logging
    val_loss_every : int = 125  # evaluate val loss every N steps (0 = only at the end)
    val_tokens     : int = 10321920  # validation tokens; keep fixed so val losses stay comparable
    save_every     : int = 0  # save a checkpoint every N steps (0 = only at the end)

    # resume: checkpoint path, first step, and train only shards sorting after skip_files
    state_dict : str = None
    start_step : int = 0
    skip_files : str = None

@dataclass
class CSCS_160GPU_2K:
    """
    160-GPU training config with 2048 context (the default for train_gpt.py).
    Tokens/step = 9600 * 2048 = 19,660,800.
    """
    # data
    input_bin     : str = "fineweb_pit/train/*.bin"
    input_val_bin : str = "fineweb_pit/val/fineweb_val_*.bin"

    # optimization
    batch_size        : int = 160 * 60
    device_batch_size : int = 4 # per-GPU micro-batch; 4 fits on a GH200
    sequence_length   : int = 2048
    learning_rate     : float = 0.0018 / 2
    warmup_iters      : int = 0
    weight_decay      : float = 0.0

    # eval/logging
    val_loss_every : int = 125
    val_tokens     : int = 13107200  # must be divisible by num_gpus * device_batch_size * sequence_length
    save_every     : int = 500

    # resume: checkpoint path, first step, and train only shards sorting after skip_files
    state_dict : str = None
    start_step : int = 0
    skip_files : str = None
