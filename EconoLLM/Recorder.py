import torch

global NET_INITTED
NET_INITTED = True
debugging = False
gpus = 2000
computationPool = [0.0] * (gpus + 1)
memoryUsage = [0.0] * (gpus + 1)
Trace = [0] * gpus
recordFile = [0] * gpus
computationFile = [0] * gpus
DetailRecord = [""] * gpus
BackwardStack = [0.0] * gpus
BackwardCommunicateStack = [[]] * gpus

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
    computationPool[rank] = 0

def WriteRecordSendrecv(type, flops, communicationFlops, groups):
    strrank = str(groups)
    rank = torch.distributed.get_rank()
    recordFile[rank].writelines([str(rank) + " " + str(type) + " " + str(flops) + " " +str(communicationFlops / communicationSpeed) + " " + str(memoryUsage[rank]) + "\n" + strrank + "\n"])   
    computationPool[rank] = 0
    
def _RecordCompute(flops):
    computationPool[torch.distributed.get_rank()] += flops / computationSpeed

def _RecordMemory(flops):
    computationPool[torch.distributed.get_rank()] += flops / memorySpeed

def clearpool():
    if computationPool[torch.distributed.get_rank()] > 0:
        WriteRecord(0, computationPool[torch.distributed.get_rank()], 0, None)
        computationPool[torch.distributed.get_rank()] = 0

def _getsize(input):
    size = 1
    for shapes in input.fakeShape:
        size *= shapes
    return size * input.elementSize
