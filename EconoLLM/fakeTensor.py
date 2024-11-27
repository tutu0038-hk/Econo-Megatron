import torch
from torch._prims_common import ShapeType
from Recorder import _RecordCompute, _RecordMemory, _getsize, memoryUsage

from torch._guards import Source
from typing import Tuple, Optional
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
)

from torch._subclasses.meta_utils import (
    MetaConverter,
)
try:
    not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")
except ValueError as e:
    if "'not_implemented' not registered" in str(e):
        import logging as not_implemented_log
    else:
        raise e
    
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

        out = (outDim, self.elementSize)
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
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
        )

FakeTensorMode.from_tensor = _from_tensor
mode = FakeTensorMode(allow_non_fake_inputs = True)
mode.fake_tensor_converter = FakeTensorConverterWithNoData()

memoryPool = {}
def FetchFakeTensor(outDim, outSize, funcName = None):
    return mode.from_tensor(outDim, outSize)

def MakeFake(self):
    if (self.__class__.__name__ == "FakeTensorWithNoData"):
        return self
    else:
        return mode.from_tensor(self.shape, self.element_size())
