# Copyright (c) OpenMMLab. All rights reserved.
import torch.distributed as dist
import torch
import random
import numpy as np
import math
from torch import Tensor
from typing import Any, Tuple


_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

_INNER_SEQUENCE_PARALLEL_GROUP = None
_INNER_SEQUENCE_PARALLEL_WORLD_SIZE = None
_INNER_SEQUENCE_PARALLEL_RANK = None

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_WORLD_SIZE = None
_DATA_PARALLEL_RANK = None


def init_sequence_parallel(sequence_parallel_size: int = 1):
    assert dist.is_initialized()
    world_size: int = dist.get_world_size()

    # enable_ds_sequence_parallel = sequence_parallel_size > 1
    # if enable_ds_sequence_parallel:
    if world_size % sequence_parallel_size != 0:
        raise RuntimeError(f'world_size ({world_size}) is not divisible by '
                           f'sequence_parallel_size {sequence_parallel_size}')

    num_sequence_parallel_groups: int = world_size // sequence_parallel_size

    rank = dist.get_rank()

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, \
        'sequence parallel group is already initialized'
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    start_rank = 0
    end_rank = world_size
    for j in range(sequence_parallel_size):
        ranks = range(start_rank + j, end_rank, sequence_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))
        group = dist.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
    

def init_sequence_parallel_seeds():
    assert dist.is_initialized()
    group_rank0 = dist.get_process_group_ranks(_SEQUENCE_PARALLEL_GROUP)[0]
    device = torch.cuda.current_device()

    if dist.get_rank() == group_rank0:
        seed = torch.randint(1000000, (1,)).to(device)
    else:
        seed = torch.zeros(1, dtype=torch.long).to(device)
    
    dist.broadcast(seed, src=group_rank0, group=_SEQUENCE_PARALLEL_GROUP)
    seed = seed.item()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_inner_sequence_parallel(inner_sequence_parallel_size: int = 1):
    """Build the sequence parallel inner groups.

    They are helpful when sp size is not evenly divided by the number of attn
    heads.
    """
    assert _SEQUENCE_PARALLEL_GROUP is not None, \
        ('Please call `init_inner_sequence_parallel` after calling '
         '`init_sequence_parallel`.')

    rank = dist.get_rank()
    world_size: int = dist.get_world_size()

    n_inner_group = world_size // inner_sequence_parallel_size

    global _INNER_SEQUENCE_PARALLEL_GROUP
    assert _INNER_SEQUENCE_PARALLEL_GROUP is None

    for i in range(n_inner_group):
        ranks = range(i * inner_sequence_parallel_size,
                      (i + 1) * inner_sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            _INNER_SEQUENCE_PARALLEL_GROUP = group


def is_inner_sequence_parallel_initialized():
    return _INNER_SEQUENCE_PARALLEL_GROUP is not None


def get_inner_sequence_parallel_group():
    return _INNER_SEQUENCE_PARALLEL_GROUP


def get_inner_sequence_parallel_world_size():
    global _INNER_SEQUENCE_PARALLEL_WORLD_SIZE
    if _INNER_SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _INNER_SEQUENCE_PARALLEL_WORLD_SIZE
    if not dist.is_initialized() or (_INNER_SEQUENCE_PARALLEL_GROUP is None):
        _INNER_SEQUENCE_PARALLEL_WORLD_SIZE = 1
    else:
        _INNER_SEQUENCE_PARALLEL_WORLD_SIZE = dist.get_world_size(
            group=get_inner_sequence_parallel_group())
    return _INNER_SEQUENCE_PARALLEL_WORLD_SIZE


def get_inner_sequence_parallel_rank():
    global _INNER_SEQUENCE_PARALLEL_RANK
    if _INNER_SEQUENCE_PARALLEL_RANK is not None:
        return _INNER_SEQUENCE_PARALLEL_RANK
    if not dist.is_initialized() or (_INNER_SEQUENCE_PARALLEL_GROUP is None):
        _INNER_SEQUENCE_PARALLEL_RANK = 0
    else:
        _INNER_SEQUENCE_PARALLEL_RANK = dist.get_rank(
            group=get_inner_sequence_parallel_group())
    return _INNER_SEQUENCE_PARALLEL_RANK


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""    
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    if not dist.is_initialized() or (_SEQUENCE_PARALLEL_GROUP is None):
        _SEQUENCE_PARALLEL_WORLD_SIZE = 1
    else:
        _SEQUENCE_PARALLEL_WORLD_SIZE = dist.get_world_size(
            group=get_sequence_parallel_group())
    return _SEQUENCE_PARALLEL_WORLD_SIZE


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    if not dist.is_initialized() or (_SEQUENCE_PARALLEL_GROUP is None):
        _SEQUENCE_PARALLEL_RANK = 0
    else:
        _SEQUENCE_PARALLEL_RANK = dist.get_rank(
            group=get_sequence_parallel_group())
    return _SEQUENCE_PARALLEL_RANK


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    global _DATA_PARALLEL_WORLD_SIZE
    if _DATA_PARALLEL_WORLD_SIZE is not None:
        return _DATA_PARALLEL_WORLD_SIZE
    if not dist.is_initialized():
        _DATA_PARALLEL_WORLD_SIZE = 1
    else:
        _DATA_PARALLEL_WORLD_SIZE = dist.get_world_size(
            group=get_data_parallel_group())
    return _DATA_PARALLEL_WORLD_SIZE


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    global _DATA_PARALLEL_RANK
    if _DATA_PARALLEL_RANK is not None:
        return _DATA_PARALLEL_RANK
    if not dist.is_initialized():
        _DATA_PARALLEL_RANK = 0
    else:
        _DATA_PARALLEL_RANK = dist.get_rank(group=get_data_parallel_group())
    return _DATA_PARALLEL_RANK


def pad_for_sequence_parallel(tensor, padding_value=0, dim=-1):
    length = tensor.shape[dim]
    seq_parallel_world_size = get_sequence_parallel_world_size()
    if length % seq_parallel_world_size == 0:
        return tensor

    pad_num = seq_parallel_world_size - (length % seq_parallel_world_size)
    pad_shape = (*tensor.shape[:dim], pad_num,
                 *tensor.shape[dim + 1:]) if dim != -1 else (
                     *tensor.shape[:dim], pad_num)
    pad = torch.full(
        pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([tensor, pad], dim=dim)
    return tensor


def split_for_sequence_parallel(input, dim: int, sp_group: dist.ProcessGroup = None, padding_value: int = 0):
    """Splits the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be split.
        dim: The dimension along which the tensor should be split.
        sp_group: The sequence parallel process group.

    Returns:
        The split tensor corresponding to the current rank's chunk.
    """
    if sp_group is None:
        sp_group = get_sequence_parallel_group()

    # automatic padding in split function
    input = pad_for_sequence_parallel(input, padding_value, dim)

    world_size = dist.get_world_size(sp_group)
    if world_size == 1:
        return input

    rank = dist.get_rank(sp_group)
    dim_size = input.size(dim)
    assert dim_size % world_size == 0, (
        f'The dimension to split ({dim_size}) is not a multiple of '
        f'world size ({world_size}), cannot split tensor evenly')

    tensor_list = torch.split(input, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


def gather_for_sequence_parallel(input, dim: int, sp_group: dist.ProcessGroup = None):
    """Gathers the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be gathered.
        dim: The dimension along which the tensor should be gathered.
        sp_group: The sequence parallel process group.

    Returns:
        The gathered tensor concatenated along the specified dimension.
    """
    if sp_group is None:
        sp_group = get_sequence_parallel_group()
        
    input = input.contiguous()
    world_size = dist.get_world_size(sp_group)
    dist.get_rank(sp_group)

    if world_size == 1:
        return input

    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    assert input.device.type != 'cpu'
    dist.all_gather(tensor_list, input, group=sp_group)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _all_to_all(
    input: Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input: Input tensor
        sp_group: Sequence parallel process group
        scatter_dim: Scatter dimension
        gather_dim: Gather dimension
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor, sp_group: dist.ProcessGroup,
                scatter_dim: int, gather_dim: int):
        ctx.sp_group = sp_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(sp_group)
        output = _all_to_all(input, ctx.world_size, sp_group, scatter_dim,
                             gather_dim)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple:
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.sp_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input: Tensor,
    sp_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    """Convenience function to apply the all-to-all operation with scatter and
    gather dimensions.

    Notes:
        We have wrapped the `torch.distributed.all_to_all` function to
        enable automatic differentiation of the all-to-all operation.

    Args:
        input: The input tensor for which all-to-all communication is performed
        sp_group: The sequence parallel process group.
        scatter_dim: The dimension along which the input tensor is scattered
            (default: 2).
        gather_dim: The dimension along which the output tensor is gathered
            (default: 1).

    Returns:
        The output tensor after the all-to-all communication.
    """
    return _AllToAll.apply(input, sp_group, scatter_dim, gather_dim)


def pre_process_for_sequence_parallel_attn(query_states,
                                           key_states,
                                           value_states,
                                           scatter_dim=2,
                                           gather_dim=1):
    b, s_div_sp, h, d = query_states.shape
    sp = get_sequence_parallel_world_size()

    if not is_inner_sequence_parallel_initialized():
        insp = sp // math.gcd(h, sp)
        init_inner_sequence_parallel(insp)
    else:
        insp = get_inner_sequence_parallel_world_size()

    def pre_process_for_inner_sp(q, k, v):
        if scatter_dim != 2 and gather_dim != 1:
            raise NotImplementedError(
                'Currently only `scatter_dim == 2` and `gather_dim == 1` '
                f'is supported. But got scatter_dim = {scatter_dim} and '
                f'gather_dim = {gather_dim}.')

        # (b, s_div_sp, h, d) ->
        # (b, s_div_sp, sp/insp, h*insp/sp, insp, d/insp) ->
        # (b, s_div_sp, sp/insp, insp, h*insp/sp, d/insp) ->
        # (b, s_div_sp, insp*h, d/insp)
        q = q.view(b, s_div_sp, sp // insp, h * insp // sp, insp,
                   d // insp).transpose(3, 4).flatten(2, 4)
        k = k.view(b, s_div_sp, sp // insp, h * insp // sp, insp,
                   d // insp).transpose(3, 4).flatten(2, 4)
        v = v.view(b, s_div_sp, sp // insp, h * insp // sp, insp,
                   d // insp).transpose(3, 4).flatten(2, 4)

        return q, k, v

    def post_process_for_inner_sp(q, k, v):
        # (b, s, insp*h/sp, d/insp) -> (b, s, insp*h/sp, d)
        q = gather_forward_split_backward(q, -1,
                                          get_inner_sequence_parallel_group())
        k = gather_forward_split_backward(k, -1,
                                          get_inner_sequence_parallel_group())
        v = gather_forward_split_backward(v, -1,
                                          get_inner_sequence_parallel_group())

        return q, k, v

    assert (h * insp) % sp == 0, \
        ('The number of attention heads should be divisible by '
         '(sequence_parallel_world_size // sequence_parallel_inner_world_size)'
         f'. But got n_head = {h}, sequence_parallel_world_size = '
         f'{sp} and sequence_parallel_inner_world_size = {insp}.')

    if insp > 1:
        query_states, key_states, value_states = pre_process_for_inner_sp(
            query_states, key_states, value_states)

    # (b, s_div_sp, insp*h, d/insp) -> (b, s, insp*h/sp, d/insp)
    sequence_parallel_group = get_sequence_parallel_group()
    query_states = all_to_all(
        query_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)
    key_states = all_to_all(
        key_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)
    value_states = all_to_all(
        value_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)

    if insp > 1:
        query_states, key_states, value_states = post_process_for_inner_sp(
            query_states, key_states, value_states)

    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output,
                                            scatter_dim=1,
                                            gather_dim=2):
    sp = get_sequence_parallel_world_size()
    insp = get_inner_sequence_parallel_world_size()
    b, s, h_mul_insp_div_sp, d = attn_output.shape
    h = h_mul_insp_div_sp * sp // insp
    s_div_sp = s // sp

    if insp > 1:
        # (b, s, insp*h/sp, d) -> (b, s, insp*h/sp, d/insp)
        attn_output = split_forward_gather_backward(
            attn_output, -1, get_inner_sequence_parallel_group())

    # (b, s, insp*h/sp, d/insp) -> (b, s_div_sp, insp*h, d/insp)
    sequence_parallel_group = get_sequence_parallel_group()
    output = all_to_all(
        attn_output,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)

    if insp > 1:
        # (b, s_div_sp, insp*h, d/insp) ->
        # (b, s_div_sp, sp/insp, insp, h*insp/sp, d/insp) ->
        # (b, s_div_sp, sp/insp, h*insp/sp, insp, d/insp) ->
        # (b, s_div_sp, h, d)
        output = output.view(b, s_div_sp, sp // insp, insp, h * insp // sp,
                             d // insp).transpose(3, 4).reshape(
                                 b, s_div_sp, h, d)

    return output


class _ReduceLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mean_loss, loss_scale, process_group):
        ctx.mode = process_group
        if loss_scale == 0:
            # convert nan to 0 just for logging
            mean_loss = torch.nan_to_num(mean_loss)
        loss_sum = mean_loss * loss_scale
        dist.all_reduce(loss_sum, group=process_group)
        dist.all_reduce(loss_scale, group=process_group)
        loss = loss_sum / loss_scale
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def reduce_sequence_parallel_loss(mean_loss,
                                  loss_scale: float = 1.0,
                                  sp_group: dist.ProcessGroup = None):
    if sp_group is None:
        sp_group = get_sequence_parallel_group()

    if dist.get_world_size(sp_group) == 1:
        return mean_loss
    if sp_group is None:
        # avoid bc breaking
        sp_group = get_sequence_parallel_group()
    return _ReduceLoss.apply(mean_loss, loss_scale, sp_group)