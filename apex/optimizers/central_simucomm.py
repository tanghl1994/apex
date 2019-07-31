import torch
import torch.distributed as dist
from .compression import *

def sam_gather(tensor,dest,group_list,async_op = False):
    
    if dist.get_rank() == dest:
        buffer_tensor = [torch.zeros_like(tensor) for i in range(dist.get_world_size())]
        buffer_tensor[dist.get_rank()].set_(tensor.data)
        for rank in range(dist.get_world_size()):
            if rank is not dist.get_rank():
                #print('=====================================================================')
                #print('dest is ',dest,' send to rank: ',rank)
                #print('=====================================================================')                
                dummy_tensor = buffer_tensor[rank]
                dist.broadcast(dummy_tensor, src = rank, group = group_list[dest][rank], async_op = async_op)
        rec_tensor = torch.zeros_like(tensor)
        for i in range(dist.get_world_size()):
            rec_tensor += buffer_tensor[i]
        tensor.set_(rec_tensor.data)

    else:
        dummy_tensor = tensor
        dist.broadcast(dummy_tensor,src = dist.get_rank(), group = group_list[dist.get_rank()][dest], async_op = async_op)



def simucentral(origin_tensor, origin_error, buffer_error, group_list, worker_schedule, async_op = False):
    w_size = dist.get_world_size()
    my_rank = dist.get_rank()
    
    tensor_list = torch.chunk(origin_tensor.clone().detach(), w_size)
    rec_tensor = tensor_list[my_rank].clone().detach()
    
    origin_error_list = torch.chunk(origin_error, w_size)
    
    for t in range(2*(w_size - 1)):
        if worker_schedule[my_rank][t][1] == 0:
            rec_pos = worker_schedule[my_rank][t][0]
            dummy_tensor = tensor_list[rec_pos]
            dist.broadcast(dummy_tensor,src = my_rank,group = group_list[my_rank][rec_pos])
        else:
            send_pos = worker_schedule[my_rank][t][0]
            dummy_tensor = torch.zeros_like(rec_tensor)
            dist.broadcast(dummy_tensor,src = send_pos,group = group_list[my_rank][send_pos])
            rec_tensor += dummy_tensor
        dist.barrier()
    tensor_list[my_rank].set_(rec_tensor)

    for src in range(w_size):
        dist.broadcast(tensor_list[src],src = src)
    
    origin_tensor.set_(torch.cat(tensor_list))


    
