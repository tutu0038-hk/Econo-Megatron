import torch
from tempfile import TemporaryDirectory
from typing import Tuple, Optional
import random
from typing import Callable
import os
from torch import nn, Tensor
import torch.distributed
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from queue import PriorityQueue
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")
except ValueError as e:
    if "'not_implemented' not registered" in str(e):
        import logging as not_implemented_log
    else:
        raise e

# from LLMsimulator_pb2 import CommunicatorInput
# from LLMsimulator_grpc import GreeterStub
# import asyncio
# from grpclib.client import Channel
import time
import torch.distributed.distributed_c10d as c10d
from torch.utils._stats import count
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, TypeVar
from torch.multiprocessing.reductions import StorageWeakRef
from weakref import ReferenceType
from torch._prims_common import ShapeType

from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    _DistributedBackendOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    PrefixStore,
    ProcessGroup,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    DebugLevel,
    get_debug_level,
    Work,
    _register_process_group,
    _resolve_process_group,
    _unregister_all_process_groups,
    _unregister_process_group,
)

global NET_INITTED
NET_INITTED = True
debugging = False
gpus = 2000
pool = [0.0] * (gpus + 1)
memoryUsage = [0.0] * (gpus + 1)
Trace = [0] * gpus
recordFile = [0] * gpus
computationFile = [0] * gpus
DetailRecord = [""] * gpus

KB = 1024
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB
TFlops = 1e12
memorySpeed = 1 * TB
communicationSpeed = 20 * GB
comunicationLatency = 10 * 1e-6
computationSpeed = 83 * TFlops
memory = 24 * GB

class records:
    type : int
    flops: float
    communicationFlops : float
    ranks : tuple

    def __init__(self, _type , _flops, _communicationFlops, _ranks):
        self.type = _type
        self.flops = _flops
        self.communicationFlops = _communicationFlops
        self.ranks = tuple(_ranks)

def WriteRecord(type, flops, communicationFlops, groups):
    if groups == None:
        ranks = [1]
    else:
        ranks = torch.distributed.get_process_group_ranks(groups)
    strrank = ""
    for item in ranks:
        strrank += str(item) + " "
    rank = torch.distributed.get_rank()
    recordFile[rank].writelines([str(rank) + " " + str(type) + " " + str(flops) + " " +str(communicationFlops / communicationSpeed) + " " + str(memoryUsage[rank]) + "\n" + strrank + "\n"])
    pool[rank] = 0

def WriteRecordSendrecv(type, flops, communicationFlops, groups):
    strrank = str(groups)
    rank = torch.distributed.get_rank()
    recordFile[rank].writelines([str(rank) + " " + str(type) + " " + str(flops) + " " +str(communicationFlops / communicationSpeed) + " " + str(memoryUsage[rank]) + "\n" + strrank + "\n"])   
    pool[rank] = 0
    
def _RecordCompute(flops):
    pool[torch.distributed.get_rank()] += flops / computationSpeed

def _RecordMemory(flops):
    pool[torch.distributed.get_rank()] += flops / memorySpeed

def clearpool():
    if pool[torch.distributed.get_rank()] > 0:
        WriteRecord(0, pool[torch.distributed.get_rank()], 0, None)
        pool[torch.distributed.get_rank()] = 0

def _getsize(input):
    size = 1
    for shapes in input.fakeShape:
        size *= shapes
    return size * input.elementSize

backupReshape = torch.Tensor.reshape
backupView = torch.Tensor.view

from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
)

from torch._subclasses.meta_utils import (
    assert_eq,
    assert_metadata_eq,
    is_sparse_any,
    is_sparse_compressed,
    MetaConverter,
)

from torch._guards import Source

class P2POpReplace:
    """
    A class to build point-to-point operations for ``batch_isend_irecv``.

    This class builds the type of P2P operation, communication buffer, peer rank,
    Process Group, and tag. Instances of this class will be passed to
    ``batch_isend_irecv`` for point-to-point communications.

    Args:
        op (Callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``torch.distributed.isend`` or
            ``torch.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int): Destination or source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with recv.
    """

    def __init__(self, op: Callable, tensor: torch.Tensor, peer: int,
                 group: Optional[ProcessGroup] = None, tag: int = 0):
        """Init."""
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag

    def __new__(cls, op: Callable, tensor: torch.Tensor, peer: int,
                group: Optional[ProcessGroup] = None, tag: int = 0):
        """Create and return a new instance of the class."""
        return object.__new__(cls)


class FakeTensorWithNoData(torch.Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """
    fakeShape: ShapeType
    fake_device: torch.device
    fake_mode: "FakeTensorMode"
    elementSize : int
    gradientSize : int
    allocateNewTensor : bool
#   constant: Optional[torch.Tensor]

    # This memorizes the unbacked SymInt representing the number of nonzero
    # elements in this tensor.  This is helpful if you do something like
    # x[mask] and y[mask]; mask.nonzero() gets repeatedly called and should
    # give a consistent unbacked SymInt.  It needs to be invalidated in the
    # same way constant is.
    # TODO: Generalize this as needed, e.g., into a trie of memos
    #_nonzero_memo: Optional[torch.SymInt]
    #_nonzero_memo_vc: Optional[int]

    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FAKE
    
    # @property
    # def nonzero_memo(self):
    #     if self._nonzero_memo is None:
    #         return None
    #     # Version counter based tracking isn't 100% sound but it's close
    #     # enough
    #     if self._nonzero_memo_vc != self._version:
    #         self._nonzero_memo = None
    #         return None
    #     return self._nonzero_memo
    def data(self):
        return self
    
    @property
    def device(self):
        return torch.device("cpu")

    def __getitem__(self, key):
        if isinstance(key, slice):
            input = (key, )
        else:
            input = key
        if isinstance(key, str):
            return self     
        if isinstance(key, int):
            input = (slice(key), )      
        length = len(self.fakeShape)
        outDim = [0] * length
        i = 0
        cnt = 0
        for slices in input:
            if not slices is None:
                if slices == -1:
                    outDim.pop(i)
                    length -= 1
                    cnt += 1
                else:
                    outDim[i] = len(range(*slices.indices(self.fakeShape[i])))
                    i += 1
        for j in range(i, length):
            outDim[j] = self.fakeShape[j + cnt]

        out = FakeTensorWithNoData(outDim, self.elementSize)
        return out
    
    def __mul__(self, scalar):
        return torch.matmul(self, scalar)
    
    def __setitem__(self, key, value):
        flops = 1
        for i in value.fakeShape:
            flops *= i
        _RecordMemory(flops * self.elementSize)

    def __truediv__(self, other):
        return self
    
    def __add__(self, other):
        sizes = _getsize(self) * 3
        _RecordMemory(sizes)
        return self
    
    @property
    def names(self):
        raise UnsupportedFakeTensorException(
            "torch.compile doesn't support named tensors"
        )

    def bfloat16(self):
        return self
    
    def is_floating_point(self):
        return True
    
    def float(self):
        return self
    
    def dim(self):
        return len(self.fakeShape)
    
    def size(self, dim=None):
        if dim != None:
            return self.fakeShape[dim]
        else:
            return tuple(self.fakeShape)

    def fakeShape(self):
        return self.fakeShape
    
    def t(self):
        outdim = self.fakeShape
        transDim = [outdim[1],outdim[0]]
        output = FetchFakeTensor(transDim, self.elementSize)
        return output
        
    @staticmethod
    def __new__(cls, dim, size, func, allocateNewTensor = False, prevfunc = None):
        self = super().__new__(cls)
        if type(dim) is list:
            self.fakeShape = dim
        else:
            self.fakeShape = list(dim)
        self.elementSize = size
        self.gradientSize = 0
        self.allocateNewTensor = allocateNewTensor
        if allocateNewTensor:
            memoryUsage[torch.distributed.get_rank()] += _getsize(self.fakeShape)
        self.nowFunc = func
        self.prevFunc = prevfunc
        return self

    def __del__(self):
        if self.allocateNewTensor:
            memoryUsage[torch.distributed.get_rank()] -= _getsize(self.fakeShape)
            print(torch.distributed.get_rank(), memoryUsage[torch.distributed.get_rank()])

    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def from_tensor(t, fake_mode):
        return fake_mode.from_tensor(t)

    @classmethod
    @count
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # need to handle here to avoid infinite recursion
        # see [in_kernel_invocation]
        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device

        # Because fake mode can return NotImplemented (if it sees a subclass
        # it doesn't know how to deal with), this test here is important
        # because the next dispatch after a fake mode will attempt to use
        # subclasses of tensors to dispatch, and any FakeTensor arguments
        # will be considered eligible.
        unrecognized_types = [
            t for t in types if not issubclass(t, FakeTensor) and t is not torch.Tensor
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FakeTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        fake_mode = None
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break

        assert fake_mode is not None

        # If the fake mode is already active, don't try to reapply it!
        # NotImplemented is the right thing to return here, because the
        # typical situation this can occur is if ProxyTensorMode returned a
        # NotImplemented because of a not implemented subclass; we may have
        # unluckily attempted to hit FakeTensor's dispatch first,
        # NotImplemented lets us keep chaining until we find the actual
        # subclass
        maybe_cur_fake_mode = torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.FAKE
        )
        if maybe_cur_fake_mode:
            not_implemented_log.debug(
                "FakeTensor mode already active: %s in %s",
                fake_mode,
                maybe_cur_fake_mode,
            )
            return NotImplemented

        with fake_mode:  # type: ignore[attr-defined]
            return func(*args, **kwargs)

    # @staticmethod
    # def _find_common_device(func, flat_args) -> Tuple[torch.device, bool]:
    #     # Returns: (common_device, has_scalar_only_inputs)

    #     # cpu - zero-dim tensors can be called in cuda kernels,
    #     # so overwrite the common_device if it the only existing
    #     # device comes from a cpu zero-dim tensor
    #     common_device = None
    #     has_scalar_only_inputs = False
    #     is_cpu_zero_dim = None

    #     def cpu_zero_dim(t):
    #         return t.device.type == "cpu" and t.dim() == 0

    #     def merge_devices(t):
    #         nonlocal common_device
    #         nonlocal is_cpu_zero_dim
    #         if not isinstance(t, FakeTensor):
    #             return

    #         if common_device is None:
    #             common_device = t.device
    #             is_cpu_zero_dim = cpu_zero_dim(t)
    #             return

    #         t_is_cpu_zero_dim = cpu_zero_dim(t)
    #         if t.device == common_device:
    #             if is_cpu_zero_dim:
    #                 is_cpu_zero_dim = t_is_cpu_zero_dim
    #             return

    #         # mismatching devices !
    #         # if current tensor is cpu 0 dim, defer to existing device
    #         if t_is_cpu_zero_dim:
    #             return

    #         # current device is from cpu 0 dim tensor, overwrite
    #         if is_cpu_zero_dim:
    #             common_device = t.device
    #             is_cpu_zero_dim = t_is_cpu_zero_dim
    #             return

    #         # mismatching devices of non-zero dim tensors, throw
    #         # This might be valid behavior and need to be explicitly modeled, e.g. reshape_as
    #         raise RuntimeError(
    #             recordFile"Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}"
    #         )

    #     for arg in flat_args:
    #         merge_devices(arg)

    #     # some functions that allow Python numbers to bind to Tensors
    #     # if we have failed to find a device, and we're running one of these operators,
    #     # we must have scalar only inputs
    #     if should_allow_numbers_as_tensors(func) and common_device is None:
    #         # ops with scalar only inputs always have result on cpu
    #         has_scalar_only_inputs = True
    #         common_device = torch.device("cpu")

    #     assert common_device is not None, f"Could not find common device for {func}"

    #     return common_device, has_scalar_only_inputs

    # We must handle tolist in a special way for FakeTensors here in the case
    # where tolist is called from torch dispatch for tensor subclasses.
    # Ordinarily, if a program calls .tolist compiling still works because there is
    # special handling in dynamo, but for tensor subclasses if .tolist is called
    # inside torch dispatch, the .tolist call may be directly on a FakeTensor.
    # This would result in an error since wrapper subclasses don't have storage.
    # To avoid this, we handle the FakeTensor case by (1) specializing on the size
    # of the tensor to create the output Python list, and (2) creating unbacked
    # symints for each element of the list.
    def tolist(self):
        assert self.dim() == 1, "NYI for higher dims"
        shape_env = self.fake_mode.shape_env
        out = []
        # Specialize on the length of the list
        for _ in range(self.fakeShape[0]):
            s = shape_env.create_unbacked_symint()
            # max value?
            torch._constrain_as_size(s, min=2)
            out.append(s)
        return out


# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it uses `WeakIdRef` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverterWithNoData:
    @property
    def tensor_memo(self):
        return self.meta_converter.tensor_memo

    meta_converter: MetaConverter
    constant_storage_mapping: Dict[StorageWeakRef, List[ReferenceType]]

    def __init__(self):
        self.meta_converter = MetaConverter()

        # map from to storage to corresponding constant tensors
        self.constant_storage_mapping = {}

    def add_constant_storage_mapping(self, fake_tensor):
        # when you have a constant, aliased tensor:
        # const_tensor.add_(torch.rand([1]))
        # all aliases of it must become no longer const
        assert isinstance(fake_tensor, FakeTensor) and fake_tensor.constant is not None
        weak_st = StorageWeakRef(fake_tensor.constant._typed_storage())

        # we need a map from a weak storage to all of its corresponding
        # constant tensors. python doesn't have the weak value equivalent
        # of defaultdict(list), so we are using a WeakValueDictionary as one
        if weak_st not in self.constant_storage_mapping:
            self.constant_storage_mapping[weak_st] = []
        self.constant_storage_mapping[weak_st].append(weakref.ref(fake_tensor))

    def invalidate_constant_aliases(self, tensor):
        assert not isinstance(tensor, FakeTensor)

        weak_st = StorageWeakRef(tensor._typed_storage())
        if weak_st not in self.constant_storage_mapping:
            return

        for weak_tensor_ref in self.constant_storage_mapping[weak_st]:
            ten = weak_tensor_ref()
            if ten is not None:
                ten._fix_weakref()
                ten.constant = None

        del self.constant_storage_mapping[weak_st]

    def _get_memo(self, t):
        tid = self.meta_converter.describer.lookup_tensor.get(t)
        if tid is None:
            return None
        return self.tensor_memo.get(tid)

    def set_tensor_memo(self, t, v):
        tid = self.meta_converter.describer.get_tensor_id(t)
        self.meta_converter.tensor_memo[tid] = v

    # You can have a real tensor that you need to convert into a fake tensor.
    # If you have a meta tensor already, call from_meta_and_device.
    #
    # You're allowed to pass a meta tensor to be turned into a fake
    # tensor; although an odd thing to do, this can occur if you're doing
    # cross ref testing and the inner test is already operating on meta tensors.
    def from_real_tensor(
        self,
        fake_mode,
        basicDim,
        basicSize,
        funcName,
        shape = None,
        make_constant=False,
        shape_env=None,
        *,
        source=None,
        symbolic_context=None,
    ):
        out = FakeTensorWithNoData(
                basicDim,
                basicSize,
                funcName,
            )
        return out

    # If you specify the device, it MUST be a meta tensor.
    def from_meta_and_device(self, fake_mode, t, device):
        assert (
            t.device.type == "meta"
        ), f"tensor's device must be `meta`, got {t.device.type} instead"
        # This is a bit abusive (this is not the "real" tensor) but whatever,
        # the meta tensor should be fresh so there's no way to get it wrong
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensorWithNoData(fake_mode, t, device)
        self.set_tensor_memo(t, out)
        return out
    
def _from_tensor(
        self,
        basicDim,
        basicSize,
        funcName = None,
        *,
        static_shapes=None,
        source: Optional[Source] = None,
        symbolic_context=None,
    ):
        shape_env = self.shape_env
        if static_shapes is None:
            static_shapes = self.static_shapes
        if static_shapes:
            assert (
                symbolic_context is None
            ), "cannot set both static_shapes and symbolic_context"
            shape_env = None
        return self.fake_tensor_converter.from_real_tensor(
            self,
            basicDim,
            basicSize,
            funcName, 
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
        )

FakeTensorMode.from_tensor = _from_tensor
mode = FakeTensorMode(allow_non_fake_inputs = True)
mode.fake_tensor_converter = FakeTensorConverterWithNoData()

memoryPool = {}
def FetchFakeTensor(outDim, outSize, funcName = None):
    return mode.from_tensor(outDim, outSize, funcName)

def MakeFake(self):
    if (self.__class__.__name__ == "FakeTensorWithNoData"):
        return self
    else:
        return mode.from_tensor(self.shape, self.element_size())

def _flatten(self, start_dim = 0, end_dim = -1):
    shape = self.fakeShape
    if end_dim == -1:
        end_dim = len(shape)
    new_len = len(shape) - (end_dim - start_dim) + 1
    outDim = [0] * new_len
    total = 1
    for i in range(start_dim, end_dim):
        total *= shape[i]
    for i in range(start_dim):
        outDim[i] = shape[i]
    outDim[start_dim] = total
    self.fakeShape = outDim
    return self

def _Linear(input: Tensor, weight, bias: Optional[Tensor] = None, 
    scale: Optional[float] = None, zero_point: Optional[int] = None):
    weight = MakeFake(weight)
    dim = len(input.fakeShape)
    outDim = [0] * dim
    totalDim = 1
    for i in range(dim):
        outDim[i] = input.fakeShape[i]
        totalDim *= outDim[i]
    len2 = len(weight.fakeShape) 
    flops = totalDim * weight.fakeShape[len2 - 2]
    totalDim = totalDim / outDim[dim - 1] * weight.fakeShape[len2 - 2]
    outDim[dim - 1] = weight.fakeShape[len2 - 2]
    flops *= 2 # constant
    output = FetchFakeTensor(outDim, input.elementSize)
    sizes = _getsize(input) + _getsize(weight) + _getsize(output)
    if bias != None:
        flops += totalDim #Ax + b
        sizes += _getsize(bias)        
    output.gradientSize = input.gradientSize + weight.gradientSize + _getsize(output)
    _RecordCompute(flops)
    _RecordMemory(sizes)
    DetailRecord[torch.distributed.get_rank()] += "Linear" + input.fakeShape + " " + weight.FakeShape + "\n"
    return output

def _matmul(tensorA, tensorB):
    if type(tensorB) is int:
        return tensorA
    if type(tensorA) is int:
        return tensorB
    tensorA = MakeFake(tensorA)
    tensorB = MakeFake(tensorB)
    dim1 = len(tensorA.fakeShape)
    dim2 = len(tensorB.fakeShape)
    outDim = [0] * dim1
    flops = 1
    for i in range(dim1):
        outDim[i] = tensorA.fakeShape[i]
        flops *= tensorA.fakeShape[i]
    flops *= tensorB.fakeShape[dim2 - 1]
    flops *= 2 #constant
    outDim[dim1 - 1] = tensorB.fakeShape[dim2 - 1]
    output = FetchFakeTensor(outDim, tensorA.elementSize)
    sizes = _getsize(tensorA) + _getsize(tensorB) + _getsize(output)
    output.gradientSize = tensorA.gradientSize + tensorB.gradientSize + _getsize(output)
    _RecordCompute(flops)    
    _RecordMemory(sizes)
    return output

def _softmax(self, dim = None, _stacklevel=5):
    sizes = _getsize(self) * 2 #load and save
    _RecordMemory(sizes)
    return self

def _dropout(input, p=0.5, training=True, inplace=False):
    return input

def _embeddings(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    weight = MakeFake(weight)
    dim = len(input.fakeShape)
    outDim = [0] * dim
    totalDim = 1
    for i in range(dim):
        outDim[i] = input.fakeShape[i]
        totalDim *= outDim[i]
    len2 = len(weight.fakeShape) 
    flops = totalDim * weight.fakeShape[len2 - 2]
    totalDim = totalDim / outDim[dim - 1] * weight.fakeShape[len2 - 2]
    outDim[dim - 1] = weight.fakeShape[len2 - 2]
    flops *= 2
    output = FetchFakeTensor(outDim, weight.elementSize)
    output.gradientSize = input.gradientSize + weight.gradientSize + _getsize(output)
    sizes = _getsize(weight) + _getsize(input) + _getsize(output)
    _RecordCompute(flops)
    _RecordMemory(sizes)
    return output

def _baddbmm(input, batch1, batch2, beta=1, alpha=1, out=None):
    #to be done
    return input

def _view_as_real(self):
    self.fakeShape.pop()
    return self

def _view_as_complex(self):
    self.fakeShape.append(2)
    return self 

def _view_as(self, other):
    newShape = []
    for shape in other.fakeShape:
        newShape.append(shape)
    self.fakeShape = newShape
    return self

#def _set_default_type(type):
#    default_type = type

def _reshape(self, *fakeShape: ShapeType):
    if (self.__class__.__name__ == "FakeTensorWithNoData"):
        allShapes = []
        for i in fakeShape:
            if type(i) is tuple:
                for x in i:
                    allShapes.append(x)
            else:
                allShapes.append(i)
        default = True
        for shapes in allShapes:
            if shapes == -1:
                default = False
        if default:
            dim = len(allShapes)
            self.fakeShape = [0] * dim
            for i in range(dim):
                self.fakeShape[i] = allShapes[i]
        else:
            totalDim = 1
            for shapes in self.fakeShape:
                totalDim *= shapes
            dim = len(allShapes)
            self.fakeShape = [0] * dim
            for i in range(dim):
                if allShapes[i] != -1:
                    totalDim /= allShapes[i]
                    self.fakeShape[i] = allShapes[i]
            for i in range(dim):
                if allShapes[i] == -1:
                    self.fakeShape[i] = totalDim
        return self
    else:
        return backupReshape(self, fakeShape)

def _view(self, *fakeShape: ShapeType):
    if (self.__class__.__name__ == "FakeTensorWithNoData"):
        allShapes = []
        for i in fakeShape:
            if type(i) is tuple:
                for x in i:
                    allShapes.append(x)
            else:
                allShapes.append(i)
        default = True
        for shapes in allShapes:
            if shapes == -1:
                default = False
        #print(fakeShape, allShapes)
        if default:
            dim = len(allShapes)
            self.fakeShape = [0] * dim
            for i in range(dim):
                self.fakeShape[i] = allShapes[i]
            #print(dim)
        else:
            totalDim = 1
            for shapes in self.fakeShape:
                totalDim *= shapes
            dim = len(allShapes)
            self.fakeShape = [0] * dim
            for i in range(dim):
                if allShapes[i] != -1:
                    totalDim /= allShapes[i]
                    self.fakeShape[i] = allShapes[i]
            for i in range(dim):
                if allShapes[i] == -1:
                    self.fakeShape[i] = totalDim
        return self
    else:
        return backupView(self, fakeShape)

def _expand(self, *fakeShape: ShapeType):
    allShapes = []
    #print("_expand", self.fakeShape)
    for i in fakeShape:
        if type(i) is tuple:
            for x in i:
                allShapes.append(x)
        else:
            allShapes.append(i)    
    self.fakeShape = allShapes
    #print("_expand", allShapes)
    return self

def _transpose(self, dim0, dim1):
    length = len(self.fakeShape)
    outdim = [0] * length
    for i in range(length):
        outdim[i] = self.fakeShape[i]
    outdim[dim0],outdim[dim1] = outdim[dim1],outdim[dim0]
    output = FetchFakeTensor(outdim, self.elementSize) 
    return output

def _silu(self, inplace=False):
    return self

def _sort(self, dim=-1, descending=False, stable=False, *, out=None):
    return self, self

def _cunsum(self, dim, *, dtype=None, out=None):
    return self

def _tocuda(
        self,
        memory_format=torch.preserve_format,
        process_group=None):
    self = MakeFake(self)
    output = FetchFakeTensor(self.fakeShape, self.elementSize)
    flops = 1
    for i in self.fakeShape:
        flops *= i
    _RecordMemory(flops * self.elementSize)
    return output

def _world_size(group: Optional[ProcessGroup] = None):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    return c10d._get_group_size(pg)

def _get_rank(group: Optional[ProcessGroup] = None):
    return torch.distributed.get_rank()

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
    WriteRecord(1, pool[torch.distributed.get_rank()], flops, pg)
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
    WriteRecord(2, pool[torch.distributed.get_rank()], flops, pg)
    return None

def _send(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    group_dst_rank = torch.distributed.get_group_rank(pg, dst)
    flops = _getsize(tensor)
    WriteRecordSendrecv(3, pool[torch.distributed.get_rank()], flops, group_dst_rank)
    return ()

def _recv(tensor: torch.Tensor, src: int, group: Optional[ProcessGroup] = None, tag: int = 0):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(tensor)
    group_src_rank = torch.distributed.get_group_rank(pg, src)
    WriteRecordSendrecv(4, pool[torch.distributed.get_rank()], flops, group_src_rank)
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
    WriteRecord(5, pool[torch.distributed.get_rank()], 0, pg)
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
    WriteRecord(6, pool[torch.distributed.get_rank()], flops, pg)
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
    WriteRecord(7, pool[torch.distributed.get_rank()], flops, group)
    return None

def _reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(input_list[0])
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(8, pool[torch.distributed.get_rank()], flops, pg)
    return None

def _reduce_scatter_base(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(output)
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(8, pool[torch.distributed.get_rank()], flops, pg)
    return None

def _all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    if group == None:
        pg = c10d._get_default_group()
    else:
        pg = group
    flops = _getsize(input_tensor)
    size =  c10d._get_group_size(pg)
    flops *= (size - 1) / size
    WriteRecord(2, pool[torch.distributed.get_rank()], flops, pg)
    return None

def _batch_isend_irecv(p2p_op_list):
    for p2p_op in p2p_op_list:
        p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
    return ()

def _todevice(self, *args, **kwargs):
    current_device = self.device
    current_dtype = self.dtype
    device_to = current_device
    dtype_to = current_dtype
    if len(args) == 1:
        if isinstance(args[0], torch.dtype):
            dtype_to = args[0]
        elif isinstance(args[0], torch.device):
            device_to = args[0]
        elif isinstance(args[0], (str, int)):
            device_to = torch.device(args[0])
        elif isinstance(args[0], torch.Tensor):
            dtype_to = args[0].dtype
            device_to = args[0].device
        else:
            raise RuntimeError(f"ShardedTensor.to() have wrong arguments: {args}")
    elif len(args) == 2:
        device_to, dtype_to = args
    else:
        dtype_to = kwargs.get("dtype", current_dtype)
        device_to = kwargs.get("device", current_device)

    device_to = torch.device(device_to) if isinstance(device_to, (str, int)) else device_to

    if (self.__class__.__name__ == "FakeTensorWithNoData"):
        return self
    else:
        outDim = self.shape
        self = FetchFakeTensor(outDim, self.element_size())
        flops = 1
        for i in self.fakeShape:
            flops *= i
        _RecordMemory(flops * self.element_size())
        return self

def global_fake_mode():
    torch.tensor = FakeTensorWithNoData

def _type_as(self, type):
    self.elementSize = type.elementSize
    return self

def _argmax(self, dim, keepdim = False):   
    shape = self.fakeShape
    length = len(shape)
    if dim == -1:
        dim = length - 1
    outDim = [0] * (length - 1)
    idx = 0
    for i in range(length):
        if i != dim:
            outDim[idx] = shape[i]
            idx += 1
    out = torch.ones(outDim)
    return out

def _empty_like(self):
    return self

def _cat(self, dim=0, out=None):
    length = len(self)
    oudDim = self[0].fakeShape
    oudDim[dim] *= length
    out = FetchFakeTensor(oudDim, self[0].elementSize)
    out.gradientSize = self[0].gradientSize
    return out

def _permute(self, *fakeShape):
    allShapes = []
    for i in fakeShape:
        if type(i) is tuple:
            for x in i:
                allShapes.append(x)
        else:
            allShapes.append(i)
    newlist = [0] * len(allShapes)
    for i in range(len(allShapes)):
        newlist[i] = self.fakeShape[allShapes[i]]
    self.fakeShape = newlist
    return self

def _contiguous(self):
    return self

def _empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format):
    allShapes = []
    for i in size:
        if type(i) is tuple or type(i) is list:
            for x in i:
                allShapes.append(x)
        else:
            allShapes.append(i)
    #print(allShapes)
    out = FetchFakeTensor(allShapes, 4)
    return out

def _split(self, split_size_or_sections, dim=0):
    self = MakeFake(self)
    if isinstance(split_size_or_sections, int):     
        count = int(self.fakeShape[dim] / split_size_or_sections)
        self.fakeShape[dim] = split_size_or_sections
        out = []
        for i in range(count):
            outTensor = FetchFakeTensor(self.fakeShape, self.elementSize)
            out.append(outTensor)
        return out
    else:
        print("Not implemented warning !!!", split_size_or_sections)
        return None

def _Parameter(self):
    out = MakeFake(self)
    sizes = _getsize(out)
    _RecordMemory(sizes)
    return out

def _normal(self, mean, std, generator=None):
    return self

from torch.nn.parameter import Parameter

def _clone(self):
    out = FetchFakeTensor(self.fakeShape, self.elementSize)
    return out

def _detach(self):
    return self

def _backward(input, grad_tensors = None):
    rank = torch.distributed.get_rank()
    WriteRecord(9, pool[rank], 0, None)
    print(input.funcName)
    
    if hasattr(input.funcName, "backward"):
        output = input.funcName.backward(input)
    else:
        _RecordCompute(input.GradientSize)
        output = _backward(input.prevFunc, grad_tensors)
    total = pool[rank]
    WriteRecord(9, pool[rank] + total / computationSpeed, 0, None)
    return None

def _apply(self, input):
    print("jiashu, backward detected")
    if hasattr(self, "backward"):
        rank = torch.distributed.get_rank()
        DetailRecord[rank].append((self, input))
    else:
        self.forward(input)

def _synchronize():
    pass

def _retain_grad(self):
    pass

def init(rank0, world_size0):
    memorySpeed = 3.35 * TB
    communicationSpeed = 900 * GB
    comunicationLatency = 10 * 1e-6
    computationSpeed = 1979 * TFlops
    memory = 80 * GB

    filename = os.getcwd() + "/result2/p%d.txt" % rank0
    recordFile[rank0] = open(filename, "w")
    
    filename = os.getcwd() + "/result2/p%d.txt" % rank0
    recordFile[rank0] = open(filename, "r")

    print("rank = ", rank0, "world_size = ", world_size0)
    nn.functional.linear = _Linear
    torch.matmul = _matmul
    nn.functional.embedding = _embeddings
    torch.view_as_complex = _view_as_complex
    torch.view_as_real = _view_as_real
    torch.Tensor.reshape = _reshape
    torch.Tensor.view = _view
    torch.Tensor.view_as = _view_as
    torch.Tensor.type_as = _type_as
    torch.Tensor.flatten = _flatten
    torch.Tensor.expand = _expand
    torch.Tensor.transpose = _transpose
    torch.Tensor.normal_ = _normal
    torch.Tensor.permute = _permute
    torch.Tensor.clone = _clone
    torch.Tensor.detach = _detach
    nn.functional.softmax = _softmax
    nn.functional.silu = _silu
    torch.softmax = _softmax
    torch.sort = _sort
    torch.cumsum = _cunsum
    torch.argmax = _argmax
    torch.empty_like = _empty_like
    torch.cat = _cat
    torch.Tensor.contiguous = _contiguous
    torch.baddbmm = _baddbmm
    torch.empty = _empty
    torch.split = _split
    #torch.Tensor = new_tensor
    torch.nn.parameter = _Parameter
    torch.nn.parameter.Parameter = _Parameter
    torch.nn.functional.dropout = _dropout
    torch.bmm = _matmul

    #torch.set_default_tensor_type = _set_default_type
    #torch.distributed.get_world_size =  _world_size
    #torch.distributed.get_rank = _get_rank
    #torch.distributed.get_group_rank = _get_rank

    #cost data transforming time
    torch.Tensor.to = _todevice
    torch.Tensor.cpu = _tocuda
    torch.Tensor.cuda = _tocuda
    
    #commnication part
    torch.distributed.all_gather = all_gather_md
    torch.distributed.all_reduce = all_reduce_md
    torch.distributed.broadcast = _broadcast
    torch.distributed.barrier = barrier_md
    torch.distributed.send = _send
    torch.distributed.recv = _recv
    torch.distributed.all_to_all = _all_to_all
    torch.distributed.reduce_scatter = _reduce_scatter
    torch.distributed._all_gather_base = _all_gather_into_tensor
    torch.distributed._reduce_scatter_base = _reduce_scatter_base
    torch.distributed.batch_isend_irecv = _batch_isend_irecv
    torch.distributed.irecv = _recv
    torch.distributed.isend = _send
    torch.distributed.P2POp = P2POpReplace
    torch.cuda.synchronize = _synchronize

    ##
    torch.autograd.backward = _backward
    torch.Tensor.retain_grad = _retain_grad
    torch.nn.Module.apply = _apply

def solve():
    print("____________solving result____________")
    print("______________________________________")
    gpus = torch.distributed.get_world_size()
    status = [gpus] * gpus
    ## status = gpus : running
    ## +x : thread waiting for send or recv x
    ## -x : thread is barried by x type operation
    times = [0] * gpus
    pretimes = [0] * gpus
    ## running time for all gpus
    index = [0] * gpus
    ## processing recordID now
    q = PriorityQueue()

    for rank in range(gpus):
        filename = os.getcwd() + "/result2/p%d.txt" % rank
        recordFile[rank] = open(filename, "r")

    # filename = os.getcwd() + "/result/Trace.txt"
    # output = open(filename, "w")
    def _insert(rank, index):
        line = recordFile[rank].readline()
        if line:
            string = str.split(line)
            types = int(string[1])
            flops = float(string[2])
            comni = float(string[3])
            line = recordFile[rank].readline()
            string = str.split(line)
            ranks = []
            for i in string:
                ranks.append(int(i))
            Trace[rank] = records(types, flops, comni, ranks)
            q.put((times[rank] + Trace[rank].flops, [rank, index + 1]))

    for i in range(gpus):
        _insert(i, 0)

    totalTime = 0
    printTrace = True
    while q.empty() == False:
        realTime, now = q.get()
        rank, i = now
        Record = Trace[rank]
        index[rank] = i
        pretimes[rank] = realTime

        if totalTime < realTime and printTrace:
            print(totalTime)
            print(times)
            print(status)
            gpuStatus = ""
            for i in range(gpus):
                gpuStatus += "GPU %d: " % i
                if status[i] == gpus:
                    gpuStatus += "Running, "
                else:
                    gpuStatus += "Waiting, "
            print(gpuStatus)
            totalTime = realTime
            msd = input("")

        if Record.type == 1 or Record.type == 2 or Record.type == 6 or Record.type == 7 or Record.type == 8: 
            #all_gather or all_reduce or broadcast etc. all these operation should wait for the process in the group
            status[rank] = -Record.type
            done = True
            for globalrank in Record.ranks:
                if status[globalrank] != -Record.type: ##some gpu in this group is still running
                    done = False
                    break
            if done: #if all groups met barrier, continue running
                times[rank] = realTime + Record.communicationFlops
                for globalrank in Record.ranks:
                    status[globalrank] = gpus
                    times[globalrank] = times[rank]
                    _insert(globalrank, index[globalrank])
        elif Record.type == 0:
            times[rank] = realTime
            _insert(rank, index[rank])
        elif Record.type == 3 or Record.type == 4: ##send and recv
            dstRecv = Record.ranks[0]
            if status[dstRecv] == gpus:
                status[rank] = dstRecv
                times[rank] = realTime
            elif status[dstRecv] == rank:
                times[rank] = realTime + Record.communicationFlops
                times[dstRecv] = times[rank]
                status[rank] = gpus
                status[dstRecv] = gpus
                _insert(rank, i)
                _insert(dstRecv, index[dstRecv])
            else:
                status[rank] = dstRecv
                times[rank] += Record.flops
        elif Record.type == 5: ##barrier
            status[rank] = -Record.type
            done = True
            for globalrank in Record.ranks:
                if status[globalrank] != -Record.type: ##some gpu in this group is still running
                    done = False
                    break
            if done: #if all groups met barrier, continue running
                times[rank] += Record.flops
                for globalrank in Record.ranks:
                    status[globalrank] = gpus
                    times[globalrank] = times[rank]
                    _insert(globalrank, index[globalrank])
        elif Record.type == 9:
            times[rank] = realTime
            _insert(rank, index[rank])
            print("Backward", rank)
        else:
            pass
    print(times)

def print_trace(rank):
    return
    cnt = 0
    for records in Trace[rank]:
        recordFile.writelines([str(rank), str(records.type), str(records.flops), str(records.communicationFlops), str(records.ranks)])
        
if __name__ == "__main__":
    solve()