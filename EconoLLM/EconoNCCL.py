import torch
from Recorder import _getsize, WriteRecord, WriteRecordSendrecv, computationPool
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Optional

from torch._C._distributed_c10d import (
    ProcessGroup,
    ReduceOp,
)

def all_reduce_md(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if async_op:
        flops = 0
    else:
        flops = _getsize(tensor)
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    size = c10d._get_group_size(pg)
    flops *= 2 * (size - 1) / size
    WriteRecord(1, computationPool[torch.distributed.get_rank()], flops, pg)
    return None

def all_gather_md(tensor_list, tensor, group=None, async_op=False):
    if async_op:
        flops = 0
    else:
        flops = _getsize(tensor)
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    size = c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(2, computationPool[torch.distributed.get_rank()], flops, pg)
    return None

def _send(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    group_dst_rank = torch.distributed.get_group_rank(pg, dst)
    flops = _getsize(tensor)
    WriteRecordSendrecv(3, computationPool[torch.distributed.get_rank()], flops, group_dst_rank)
    return ()

def _recv(tensor: torch.Tensor, src: int, group: Optional[ProcessGroup] = None, tag: int = 0):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(tensor)
    group_src_rank = torch.distributed.get_group_rank(pg, src)
    WriteRecordSendrecv(4, computationPool[torch.distributed.get_rank()], flops, group_src_rank)
    return ()

def barrier_md(group=None, async_op=False, device_ids=None):
    """
    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().
    """
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    WriteRecord(5, computationPool[torch.distributed.get_rank()], 0, pg)
    return None

def _all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = 0
    for tensor in input_tensor_list:
        flops +=_getsize(tensor)
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(6, computationPool[torch.distributed.get_rank()], flops, pg)
    return None

def _broadcast(tensor, src, group=None, async_op=False):
    if (tensor.__class__.__name__ != "FakeTensorWithNoData"):
        return
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(tensor)
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(7, computationPool[torch.distributed.get_rank()], flops, group)
    return None

def _reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(input_list[0])
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(8, computationPool[torch.distributed.get_rank()], flops, pg)
    return None

def _reduce_scatter_base(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(output)
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(8, computationPool[torch.distributed.get_rank()], flops, pg)
    return None

def _all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(input_tensor)
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(2, computationPool[torch.distributed.get_rank()], flops, pg)
    return None

def _batch_isend_irecv(p2p_op_list):
    for p2p_op in p2p_op_list:
        p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
    return ()