import torch.distributed as dist
import torch
import argparse
#from temp_optim import simucomm
from newtestsimucomm import *
import torch.multiprocessing as mp

def sam_gather(tensor,dest,group_list,async_op = False): 
    if dist.get_rank() == dest:
        buffer_tensor = [torch.zeros_like(tensor) for i in range(dist.get_world_size())]
        buffer_tensor[dist.get_rank()].set_(tensor.data)
        for rank in range(dist.get_world_size()):
            if rank is not dist.get_rank():
                #print('dest is ',dest,' send to rank: ',rank)
                dummy_tensor = buffer_tensor[rank]
                dist.broadcast(dummy_tensor, src = rank, group = group_list[dest][rank], async_op = async_op)
                #if dist.get_rank() == 0:
                 #   print('Getting ',buffer_tensor[rank], ' from worker ', rank)
        rec_tensor = torch.zeros_like(tensor)
        for i in range(dist.get_world_size()):
            rec_tensor += buffer_tensor[i]
        tensor.set_(rec_tensor.data)
        #if dist.get_rank() == 0:
          #  print(buffer_tensor)
          #  print(rec_tensor)
        
    else:
        dummy_tensor = tensor
        dist.broadcast(dummy_tensor,src = dist.get_rank(), group = group_list[dist.get_rank()][dest], async_op = async_op) 

def sam_central(tensor,group_list):
    w_size = dist.get_world_size()
    tensor_list = torch.chunk(tensor.clone().detach(), w_size)
    for rank in range(w_size):
        sam_gather(tensor_list[rank],dest = rank,group_list = group_list, async_op = False)
        dist.barrier()
    if dist.get_rank() == 0:
        print(tensor_list[0])
    for src in range(w_size): 
        dist.broadcast(tensor_list[src],src = src, async_op = False)
    tensor.set_(torch.cat(tensor_list))



def simucentral(tensor,group_list):
    w_size = dist.get_world_size()
    rank = dist.get_rank()
    
    s_rec = torch.cuda.Stream()
    s_send = torch.cuda.Stream()

    tensor_list = torch.chunk(tensor,w_size)
    rec_tensor = torch.zeros_like(tensor_list[rank])

    #for t in range(w_size - 1):
       # print('t is : ', t)
    t = 0    
    with torch.cuda.stream(s_send):
        send_pos = (rank + t + 1)%w_size
        print('Worker ', rank, 'Send to  ',send_pos)
        dummy_tensor = tensor_list[send_pos].clone().detach()
        #group = dist.new_group(ranks = [rank,send_pos])
        dist.broadcast(dummy_tensor, src = rank, group = group_list[rank][send_pos], async_op = True)
    with torch.cuda.stream(s_rec):
        rec_pos = (rank - t - 1)%w_size
        print('Worker ', rank, 'Receive from  ',rec_pos)
        dummy_tensor = torch.zeros_like(rec_tensor)
        #group = dist.new_group(ranks = [rank,rec_pos])
        dist.broadcast(dummy_tensor, src = rec_pos, group = group_list[rank][rec_pos], async_op = True)
        rec_tensor += dummy_tensor
    print('OK I am Here')
    torch.cuda.synchronize()

    for idx in range(w_size):
        if dist.get_rank() == idx:
            dummy_tensor = rec_tensor
            dist.broadcast(dummy_tensor, src = idx)
        else:
            dummy_tensor = torch.zeros_like(tensor_list[idx])
            dist.broadcast(dummy_tensor, src = idx)


parser = argparse.ArgumentParser()
# Required_parameter
parser.add_argument("--local_rank",
                    type=int,
                    default=0,
                    help="local_rank for distributed training on gpus")

#dist.init_process_group(backend='nccl')
def testsimu(gpu,w_size):
    dist.init_process_group(init_method='tcp://127.0.0.1:8008',backend='nccl',rank=gpu,world_size=w_size)
    print(dist.get_world_size())                    
    #w_size = dist.get_world_size()
    group_list = [dist.new_group(ranks = [i,(i+1)%dist.get_world_size()]) for i in range(dist.get_world_size())]
    '''group_list = [ [ [] for j in range(w_size) ] for i in range(w_size)]
    for i in range(dist.get_world_size() - 1): 
        for j in range(dist.get_world_size() - i - 1): 
        #if dist.get_rank() == 0:
         #   print('Creating group ',j,' to ',(i+j+1))
            group = dist.new_group(ranks = [i+j+1,j])
            group_list[j][j + i + 1] = group
            group_list[j + i + 1][j] = group'''

    args = parser.parse_args()
    device = torch.device("cuda", gpu)

    dim = 10 * dist.get_world_size()
    a = (torch.zeros(dim) + dist.get_rank() + 1).to(device)
    if dist.get_rank() == 0:
        print('a is:  ', a[0:10])
    b = a.clone().detach()
    temp1 = torch.zeros_like(a)
    temp2 = torch.zeros_like(a)
    simusend(a,temp1,temp2,group_list)
    dist.all_reduce(b)
    b /= w_size
    if dist.get_rank() == 0:
        print('My algorithm output is: ', a[0:10])
        print('AllReduce output is: ', b[0:10])

if __name__ == '__main__':
    w_size = 4
    mp.spawn(testsimu,nprocs=w_size,args=([w_size]))

        












            
