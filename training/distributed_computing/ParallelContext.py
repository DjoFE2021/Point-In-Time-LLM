from dataclasses import dataclass
import os
import torch.distributed as dist


@dataclass
class ParallelContext:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int

    tp_size: int
    dp_size: int

    tp_group: object | None
    dp_group: object | None

    tp_group_ranks: list[int]
    dp_group_ranks: list[int]

    tp_rank: int
    dp_rank: int


def _get_env_int(name: str, default: int | None = None) -> int:
    value = os.environ.get(name, None)
    if value is None:
        if default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    return int(value)


def choose_tp_size(tp_arg: str, world_size: int, local_world_size: int) -> int:
    """
    Decide TP size.

    Rules:
    - explicit integer wins
    - 'auto' picks a conservative node-local TP
    - TP must divide both world_size and local_world_size
    """
    if tp_arg != "auto":
        tp_size = int(tp_arg)
    else:
        # Conservative default:
        # start with TP=2 when possible, otherwise TP=1
        tp_size = 2 if local_world_size >= 2 else 1

    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")

    if world_size % tp_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tp_size={tp_size}"
        )

    # We want TP groups to stay within a node.
    if local_world_size % tp_size != 0:
        raise ValueError(
            f"local_world_size={local_world_size} must be divisible by tp_size={tp_size} "
            f"to keep TP groups node-local"
        )

    return tp_size


def build_parallel_context(tp_arg: str = "1") -> ParallelContext:
    """
    Build TP and DP process groups from torchrun environment.

    Assumptions:
    - one process per GPU
    - homogeneous LOCAL_WORLD_SIZE across nodes
    - ranks are assigned contiguously within each node, which is torchrun's usual layout

    Example for 2 nodes x 4 GPUs = 8 ranks, tp_size=2:

      Node 0 ranks: [0,1,2,3]
      Node 1 ranks: [4,5,6,7]

      TP groups (node-local contiguous pairs):
        [0,1], [2,3], [4,5], [6,7]

      DP groups (same shard position across replicas):
        [0,2,4,6], [1,3,5,7]
    """
    if not dist.is_initialized():
        raise RuntimeError("dist.init_process_group() must be called before build_parallel_context()")

    rank = _get_env_int("RANK")
    world_size = _get_env_int("WORLD_SIZE")
    local_rank = _get_env_int("LOCAL_RANK")
    local_world_size = _get_env_int("LOCAL_WORLD_SIZE", default=1)

    tp_size = choose_tp_size(tp_arg, world_size, local_world_size)
    dp_size = world_size // tp_size

    # Figure out which node this rank belongs to
    node_id = rank // local_world_size
    num_nodes = world_size // local_world_size

    tp_group = None
    dp_group = None
    tp_group_ranks = None
    dp_group_ranks = None

    # -------------------------
    # Build TP groups
    # -------------------------
    # Within each node, partition local ranks into contiguous chunks of size tp_size.
    # Example on a 4-GPU node with tp_size=2:
    # node ranks [0,1,2,3] -> TP groups [0,1] and [2,3]
    tp_groups_per_node = local_world_size // tp_size

    for n in range(num_nodes):
        node_base = n * local_world_size
        for chunk in range(tp_groups_per_node):
            ranks = list(range(
                node_base + chunk * tp_size,
                node_base + (chunk + 1) * tp_size
            ))
            g = dist.new_group(ranks=ranks)
            if rank in ranks:
                tp_group = g
                tp_group_ranks = ranks

    if tp_group is None:
        raise RuntimeError("Failed to assign TP group")

    # -------------------------
    # Build DP groups
    # -------------------------
    # DP groups connect identical TP positions across ALL TP replicas
    # (both cross-node and cross-chunk within a node).
    #
    # Example:
    # world_size=8, local_world_size=4, tp_size=2
    # TP groups: [0,1], [2,3], [4,5], [6,7]
    # DP groups: [0,2,4,6], [1,3,5,7]
    #
    # Rank 0 (slot 0 of [0,1]) must sync with rank 2 (slot 0 of [2,3]),
    # rank 4 (slot 0 of [4,5]), and rank 6 (slot 0 of [6,7]).
    # They all own the same weight shard and process independent micro-batches.
    #
    # General construction:
    # For each TP slot k in [0, tp_size), collect that slot from every
    # TP group across all nodes and all chunks within a node.
    for tp_slot in range(tp_size):
        ranks = []
        for n in range(num_nodes):
            node_base = n * local_world_size
            for chunk in range(tp_groups_per_node):
                ranks.append(node_base + chunk * tp_size + tp_slot)
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            dp_group = g
            dp_group_ranks = ranks

    if dp_group is None:
        raise RuntimeError("Failed to assign DP group")

    return ParallelContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        tp_size=tp_size,
        dp_size=dp_size,
        tp_group=tp_group,
        dp_group=dp_group,
        tp_group_ranks=tp_group_ranks,
        dp_group_ranks=dp_group_ranks,
        tp_rank=dist.get_rank(tp_group),
        dp_rank=dist.get_rank(dp_group),
    )