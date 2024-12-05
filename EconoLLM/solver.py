import torch
from EconoLLM.ReplaceTensor import recordFile, Trace, records
import os
from queue import PriorityQueue

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
    for records in Trace[rank]:
        recordFile.writelines([str(rank), str(records.type), str(records.flops), str(records.communicationFlops), str(records.ranks)])
       