import torch
from fakeTensor import MakeFake, FetchFakeTensor, FakeTensorWithNoData
from Recorder import _getsize, _RecordCompute, _RecordMemory, BackwardCommunicateStack, BackwardStack
from torch._prims_common import ShapeType
import torch.distributed.distributed_c10d as c10d

from typing import Tuple, Optional

from torch._C._distributed_c10d import (
    ProcessGroup,
)

backupReshape = torch.Tensor.reshape
backupView = torch.Tensor.view

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
    #DetailRecord[torch.distributed.get_rank()] += "Linear" + input.fakeShape + " " + weight.FakeShape + "\n"
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

def _clone(self):
    out = FetchFakeTensor(self.fakeShape, self.elementSize)
    return out

def _detach(self):
    return self

def _backward(input, grad_tensors = None):
    rank = torch.distributed.get_rank()
    while(BackwardCommunicateStack[rank].empty() == False):
        _RecordMemory(BackwardStack[rank].top())
        BackwardStack[rank].pop()
        func, input = BackwardCommunicateStack.top()
        func()
        BackwardCommunicateStack.pop()
        print(BackwardStack[rank].top())
    return None

def _apply(self, input):
    print("Econo, backward detected")
    if hasattr(self, "backward"):
        rank = torch.distributed.get_rank()
        BackwardCommunicateStack[rank].push((self, input))
        BackwardStack[rank].push(0)
    else:
        self.forward(input)

def _synchronize():
    pass

def _retain_grad(self):
    pass