from dataclasses import dataclass

@dataclass
class CSCS_4GPU_2K:
    """
    4-GPU training config with 2048 context.
    Keeps tokens/step = 491,520 (same as 480 * 1024).
    """
    # data
    input_bin     : str = "/capstor/scratch/cscs/sjohanne/fineweb_monthly_7B/*.bin"
    input_val_bin : str = "/capstor/scratch/cscs/sjohanne/fineweb1B/fineweb_val_*.bin"

    # optimization
    batch_size        : int = 16
    device_batch_size : int = 4 # GH200: 4
    sequence_length   : int = 2048
    learning_rate     : float = 0.0018 / 2
    warmup_iters      : int = 0
    weight_decay      : float = 0.0

    # eval/logging
    val_loss_every : int = 125
    val_tokens     : int = 13107200  # divisible by 160 GPUs × 2048 × 1
    save_every     : int = 500
    
    #loading from a state
    state_dict : str = None
    start_step : int = 0
    skip_files : str = None

@dataclass
class CSCS_60GPU:
    """
    Experiment/training configuration for a 60‑GPU CSCS run.

    Parameters
    ----------
    input_bin : str, optional
        Glob/path to training shards (.bin). Default is
        "/capstor/scratch/cscs/sjohanne/fineweb_monthly/CC-MAIN*.bin".
    input_val_bin : str, optional
        Glob/path to validation shards (.bin) used for loss eval.
        Default is "/capstor/scratch/cscs/sjohanne/fineweb10B/fineweb_val_*.bin".
    batch_size : int, optional
        Global batch size (sequences) across all devices. Default is ``8*60``.
    device_batch_size : int, optional
        Per‑device batch size (sequences). Default is 8.
    sequence_length : int, optional
        Tokens per sequence. Default is 1024.
    num_iterations : int, optional
        Total optimization iterations. Default is ``2*11190``.
    learning_rate : float, optional
        Peak/base learning rate. Default is ``0.0036 / 2``.
    warmup_iters : int, optional
        Number of warmup iterations at start. Default is 0.
    warmdown_iters : int, optional
        Linear decay iterations at the end of training. Default is ``2*2906``.
    weight_decay : float, optional
        L2 weight decay coefficient. Default is 0.
    val_loss_every : int, optional
        Frequency (in steps) of validation-loss evaluation; 0 = only at end.
        Default is 125.
    val_tokens : int, optional
        Number of tokens from validation set to evaluate, kept fixed for comparability.
        Default is 10321920.
    save_every : int, optional
        Frequency (in steps) to save checkpoints; 0 = only at end. Default is 0.

    Notes
    -----
    - Inline comments reflect historical tweaks (e.g. different defaults for 8 GPUs).
    - Learning-rate schedule parameters (``warmup_iters``, ``warmdown_iters``) are
      intended for use with a triangular/trapezoidal scheduler.
    """
    # data hyperparams
    input_bin : str = "/capstor/scratch/cscs/sjohanne/fineweb_monthly/*.bin"  # input .bin to train on
    input_val_bin : str = "/capstor/scratch/cscs/sjohanne/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*60  # batch size, in sequences, across all devices
    device_batch_size : int = 8  # 12 for 8 GPUs, 80 for 60 # batch size, in sequences, per device
    sequence_length : int = 1024  # sequence length, in tokens
    num_iterations : int = 65349  # 2*10172 # number of iterations to run
    learning_rate : float = 0.0036 / 2
    warmup_iters : int = 0
    warmdown_iters : int = 16337  # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125  # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10321920  # 10321920#10420224 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0  # every how many steps to save the checkpoint? 0 for only at the end
    
@dataclass
class CSCS_60GPU_2K:
    """
    Experiment/training configuration for a 60‑GPU CSCS run.

    Parameters
    ----------
    input_bin : str, optional
        Glob/path to training shards (.bin). Default is
        "/capstor/scratch/cscs/sjohanne/fineweb_monthly/CC-MAIN*.bin".
    input_val_bin : str, optional
        Glob/path to validation shards (.bin) used for loss eval.
        Default is "/capstor/scratch/cscs/sjohanne/fineweb10B/fineweb_val_*.bin".
    batch_size : int, optional
        Global batch size (sequences) across all devices. Default is ``8*60``.
    device_batch_size : int, optional
        Per‑device batch size (sequences). Default is 8.
    sequence_length : int, optional
        Tokens per sequence. Default is 1024.
    num_iterations : int, optional
        Total optimization iterations. Default is ``2*11190``.
    learning_rate : float, optional
        Peak/base learning rate. Default is ``0.0036 / 2``.
    warmup_iters : int, optional
        Number of warmup iterations at start. Default is 0.
    warmdown_iters : int, optional
        Linear decay iterations at the end of training. Default is ``2*2906``.
    weight_decay : float, optional
        L2 weight decay coefficient. Default is 0.
    val_loss_every : int, optional
        Frequency (in steps) of validation-loss evaluation; 0 = only at end.
        Default is 125.
    val_tokens : int, optional
        Number of tokens from validation set to evaluate, kept fixed for comparability.
        Default is 10321920.
    save_every : int, optional
        Frequency (in steps) to save checkpoints; 0 = only at end. Default is 0.

    Notes
    -----
    - Inline comments reflect historical tweaks (e.g. different defaults for 8 GPUs).
    - Learning-rate schedule parameters (``warmup_iters``, ``warmdown_iters``) are
      intended for use with a triangular/trapezoidal scheduler.
    """
    # data hyperparams
    input_bin : str = "/capstor/scratch/cscs/sjohanne/fineweb_monthly/*.bin"  # input .bin to train on
    input_val_bin : str = "/capstor/scratch/cscs/sjohanne/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*60  # batch size, in sequences, across all devices
    device_batch_size : int = 8  # 12 for 8 GPUs, 80 for 60 # batch size, in sequences, per device
    sequence_length : int = 1024  # sequence length, in tokens
    num_iterations : int = 65349  # 2*10172 # number of iterations to run
    learning_rate : float = 0.0036 / 2
    warmup_iters : int = 0
    warmdown_iters : int = 16337  # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125  # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10321920  # 10321920#10420224 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0  # every how many steps to save the checkpoint? 0 for only at the end
    
@dataclass
class CSCS_80GPU_2K:
    """
    60-GPU training config with 2048 context.
    Keeps tokens/step = 491,520 (same as 480 * 1024).
    """
    # data
    input_bin     : str = "/capstor/scratch/cscs/sjohanne/fineweb_monthly_7B/*.bin"
    input_val_bin : str = "/capstor/scratch/cscs/sjohanne/fineweb10B/fineweb_val_*.bin"

    # optimization
    batch_size        : int = 240      # 120 GPUs * 4 per device
    device_batch_size : int = 3
    sequence_length   : int = 2048
    learning_rate     : float = 0.0036 / 2
    warmup_iters      : int = 0
    weight_decay      : float = 0.0

    # eval/logging
    val_loss_every : int = 125
    val_tokens     : int = 10813440
    save_every     : int = 0
    
@dataclass
class CSCS_160GPU_2K:
    """
    160-GPU training config with 2048 context.
    Keeps tokens/step = 491,520 (same as 480 * 1024).
    """
    # data
    input_bin     : str = "/capstor/scratch/cscs/sjohanne/fineweb_monthly_7B/*.bin"
    input_val_bin : str = "/capstor/scratch/cscs/sjohanne/fineweb1B/fineweb_val_*.bin"

    # optimization
    batch_size        : int = 160 * 60
    device_batch_size : int = 4 # GH200: 4
    sequence_length   : int = 2048
    learning_rate     : float = 0.0018 / 2
    warmup_iters      : int = 0
    weight_decay      : float = 0.0

    # eval/logging
    val_loss_every : int = 125
    val_tokens     : int = 13107200  # divisible by 160 GPUs × 2048 × 1
    save_every     : int = 500
    
    #loading from a state
    state_dict : str = None
    start_step : int = 0
    skip_files : str = None
    
    
class CSCS_SoftPrompt:
    
    # data
    text_files : str = "/capstor/scratch/cscs/sjohanne/WSJ_token/*.parquet"
    
    #model
    model_size : str = "1B"
    model_path : str = "/capstor/scratch/cscs/sjohanne/checkpoints_7B/2025-03/GPT.pt"
    
    #training parameters
    batch_size : int = 4
    device_batch_size : int = int(batch_size/4)
    grad_accum_steps : int = int(batch_size/device_batch_size)
    num_steps : int = 500
    learning_rate : float = 3e-4
    weight_decay : float = 0.01
    warmdown_iters: int = int(num_steps/2)
    warmup_iters: int = 0
    max_grad_norm: float = 1.0
    val_steps: int = 50
    
    
