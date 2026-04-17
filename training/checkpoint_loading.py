import torch

from collections import OrderedDict

from models.GPT import GPT
from optimizers.lr_scheduler import LRScheduler

def load_checkpoint_to_model(
    ckpt : dict,
    model : GPT) -> GPT:
    
    raw_sd = ckpt["model"]
    new_sd = OrderedDict()
    for k, v in raw_sd.items():
        # remove 'module.' if present
        name = k.replace("module.", "")
        # or remove 'model.' if your dict was nested: name = name.replace("model.", "")
        new_sd[name] = v
        
    ckpt["model"] = new_sd
    
    #Load the weights in the model
    try:
        model.load_state_dict(ckpt["model"])
    except:
        new_sd = OrderedDict()
        for k, v in raw_sd.items():
            # remove 'module.' if present
            name = k.replace("module._orig_mod.", "").replace("_orig_mod.", "")

            # or remove 'model.' if your dict was nested: name = name.replace("model.", "")
            new_sd[name] = v
        
        ckpt["model"] = new_sd
        model.load_state_dict(ckpt["model"])
    
    return model  


def load_checkpoint_to_optimizers(optimizers,
                                  checkpoint : dict):
    """
    Restore a list of optimizers from a checkpoint.

    Args:
        optimizers (list): List of torch.optim.Optimizer objects (already constructed).
        checkpoint (dict): Loaded checkpoint with key "optimizers" containing states.

    Returns:
        list: The same list of optimizers, restored to the checkpoint state.
    """
    opt_states = checkpoint.get("optimizers", None)
    if opt_states is None:
        raise ValueError("Checkpoint does not contain optimizer states under key 'optimizers'.")

    if len(opt_states) != len(optimizers):
        raise ValueError(f"Mismatch: {len(opt_states)} states in checkpoint vs {len(optimizers)} optimizers provided.")

    for opt, state in zip(optimizers, opt_states):
        opt.load_state_dict(state)

    return optimizers

def build_scheduler_at_step(scheduler_cls, optimizer, last_epoch, **kwargs):
    """
    Recreate a scheduler as if it has already advanced `step_idx` times.

    Args:
        scheduler_cls: a class from torch.optim.lr_scheduler (e.g., CosineAnnealingLR)
        optimizer: the optimizer the scheduler controls
        step_idx (int): number of scheduler steps that already happened
        **kwargs: the usual scheduler constructor kwargs (T_max, milestones, etc.)

    Returns:
        A scheduler ready to continue from step_idx.
    """
    # Most _LRScheduler subclasses accept last_epoch
    if "last_epoch" in scheduler_cls.__init__.__code__.co_varnames:
        return scheduler_cls(optimizer, last_epoch=last_epoch, **kwargs)


def load_checkpoint_to_schedulers(
    checkpoint : dict,
    optimizers : list,
    scheduler_step : int,
    num_iterations : int,
    warmdown_iters : int,
    warmup_iters   : int):
    """
    Restore a list of LR schedulers from a checkpoint.
    """
    sch_states = checkpoint.get("schedulers", None)
    
    if sch_states is None:
        lr_scheduler = LRScheduler(num_iterations, warmdown_iters, warmup_iters)
        
        schedulers = [build_scheduler_at_step(torch.optim.lr_scheduler.LambdaLR, opt, scheduler_step, lr_lambda = lr_scheduler.get_lr) for opt in optimizers]

        return schedulers
    
    lr_scheduler = LRScheduler(num_iterations, warmdown_iters, warmup_iters)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_scheduler.get_lr) for opt in optimizers]

    if len(sch_states) != len(schedulers):
        raise ValueError(f"Mismatch: {len(sch_states)} states in checkpoint vs {len(schedulers)} schedulers provided.")

    for sch, state in zip(schedulers, sch_states):
        sch.load_state_dict(state)

    return schedulers
    
    
    
    