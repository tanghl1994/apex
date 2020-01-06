from .compression import *
import torch
import torch.distributed as dist

def _compute_grad_norm(self, fp16_grads_flat, norm_type=2):
        try:
            norm = float(torch.norm(fp16_grads_flat, 2.0, dtype=torch.float32))
        except TypeError as err:
            norm = float(torch.norm(fp16_grads_flat.float(), 2.0))
        if norm == float('inf') or norm == -float('inf') or norm != norm:
            #print(norm)
            return -1
        else:
            return norm

def simusend(origin_tensor, origin_error, buffer_error, send_groups):
    
    w_size = dist.get_world_size()
    
    origin_list = torch.chunk(origin_tensor,w_size)
    buffer_tensor = origin_tensor.clone().detach()    
    tensor_list = list(torch.chunk(buffer_tensor, w_size))

    ori_error_list = torch.chunk(origin_error, w_size)
    error_list = list(torch.chunk(buffer_error, w_size))
    
    scale_list = [torch.zeros_like(buffer_error[0]) for i in range(w_size)]
    sign_list = list(torch.chunk(torch.zeros_like(origin_tensor), w_size))
    
    pos = (w_size - 1 - dist.get_rank() + 1)%(w_size)

    #return 0
    #dummy_tensor = torch.zeros_like(tensor_list[0]) + dist.get_rank()
    #print('Before is: ',dummy_tensor)
    #if dist.get_rank() == 1:
     #   print(tensor_list[0][0:10])
    for t in range(w_size - 1):           
        #print('Pos is: ',pos)
        #if dist.get_rank() == 0:
         #   print('Step is : ', t, ' original vector leads to  ',tensor_list[0][0:10],' and ', error_list[0][0:10])
        scale_list[pos],sign_list[pos],diff = imple_naive_compress(tensor_list[pos] + ori_error_list[pos])
        error_list[pos].set_(diff)
        #if dist.get_rank() == 0:
         #   print('Step is : ', t, ' compression leads to  ',sign_list[0][0:10])
        #print('Scale is: ',scale_list[pos])
        #dist.broadcast(dummy_tensor, src = 0, async_op = True)
        send_src = dist.get_rank()
        rec_src = (dist.get_rank() - 1)%w_size
        
        if dist.get_rank()%2 == 0:
            dummy_scale = torch.zeros_like(scale_list[pos])
            dummy_scale.set_(scale_list[pos])
            dist.broadcast(dummy_scale, src = send_src, group = send_groups[send_src])
        else:
            dummy_scale = torch.zeros_like(scale_list[(pos + 1)%w_size])
            dist.broadcast(dummy_scale, src = rec_src, group = send_groups[rec_src])
            scale_list[(pos + 1)%w_size].set_(dummy_scale)
        if dist.get_rank()%2 == 1:
            dummy_scale = torch.zeros_like(scale_list[pos])
            dummy_scale.set_(scale_list[pos])
            dist.broadcast(dummy_scale, src = send_src, group = send_groups[send_src])
        else:
            dummy_scale = torch.zeros_like(scale_list[(pos + 1)%w_size])
            dist.broadcast(dummy_scale, src = rec_src, group = send_groups[rec_src])
            scale_list[(pos + 1)%w_size].set_(dummy_scale)
        
        if dist.get_rank()%2 == 0:
            dummy_sign = torch.zeros_like(sign_list[pos])
            dummy_sign.set_(sign_list[pos])
            dist.broadcast(dummy_sign, src = send_src, group = send_groups[send_src])
        else:
            dummy_sign = torch.zeros_like(sign_list[(pos + 1)%w_size])
            dist.broadcast(dummy_sign, src = rec_src, group = send_groups[rec_src])
            #if dist.get_rank() == 1 and t==0:
             #   print('receive from ',rec_src,' gets ',dummy_sign[0:10])
            sign_list[(pos + 1)%w_size].set_(dummy_sign)
        if dist.get_rank()%2 == 1:
            dummy_sign = torch.zeros_like(sign_list[pos])
            dummy_sign.set_(sign_list[pos])
            dist.broadcast(dummy_sign, src = send_src, group = send_groups[send_src])
        else:
            dummy_sign = torch.zeros_like(sign_list[(pos + 1)%w_size])
            dist.broadcast(dummy_sign, src = rec_src, group = send_groups[rec_src])
            sign_list[(pos + 1)%w_size].set_(dummy_sign)       
        

        pos = (pos + 1)%w_size
        tensor_list[pos] = (1 - 1/(t+2)) * scale_list[pos] * sign_list[pos].float() + (1/(t+2)) * origin_list[pos]
        '''if dist.get_rank() == 1 and t==0:
                print('After transform gets ',(2 * sign_list[pos].float() - 1)[0:10])
                print('But the origin tensor is  ',origin_list[pos][0:10])
        if dist.get_rank() == (t+1):
            print('Step is : ', t, ' leads to  ',tensor_list[0][0:10])'''
        
    #pos = (pos + 1)%w_size
    scale_list[pos],sign_list[pos],diff = imple_naive_compress(tensor_list[pos] + ori_error_list[pos])
    error_list[pos].set_(diff)
    tensor_list[pos] = scale_list[pos] * sign_list[pos].float()
    #print('step 1 finished')
    #if dist.get_rank() == 0:
     #   print(tensor_list[0][0:10])
    for t in range(w_size - 1):
        send_src = dist.get_rank()
        rec_src = (dist.get_rank() - 1)%w_size
        
        if dist.get_rank()%2 == 0:
            dummy_scale = torch.zeros_like(scale_list[pos])
            dummy_scale.set_(scale_list[pos])
            dist.broadcast(dummy_scale, src = send_src, group = send_groups[send_src])
        else:
            dummy_scale = torch.zeros_like(scale_list[(pos + 1)%w_size])
            dist.broadcast(dummy_scale, src = rec_src, group = send_groups[rec_src])
            scale_list[(pos + 1)%w_size].set_(dummy_scale)
        if dist.get_rank()%2 == 1:
            dummy_scale = torch.zeros_like(scale_list[pos])
            dummy_scale.set_(scale_list[pos])
            dist.broadcast(dummy_scale, src = send_src, group = send_groups[send_src])
        else:
            dummy_scale = torch.zeros_like(scale_list[(pos + 1)%w_size])
            dist.broadcast(dummy_scale, src = rec_src, group = send_groups[rec_src])
            scale_list[(pos + 1)%w_size].set_(dummy_scale)

        
        if dist.get_rank()%2 == 0:
            dummy_sign = torch.zeros_like(sign_list[pos])
            dummy_sign.set_(sign_list[pos])
            dist.broadcast(dummy_sign, src = send_src, group = send_groups[send_src])
        else:
            dummy_sign = torch.zeros_like(sign_list[(pos + 1)%w_size])
            dist.broadcast(dummy_sign, src = rec_src, group = send_groups[rec_src])
            sign_list[(pos + 1)%w_size].set_(dummy_sign)
        if dist.get_rank()%2 == 1:
            dummy_sign = torch.zeros_like(sign_list[pos])
            dummy_sign.set_(sign_list[pos])
            dist.broadcast(dummy_sign, src = send_src, group = send_groups[send_src])
        else:
            dummy_sign = torch.zeros_like(sign_list[(pos + 1)%w_size])
            dist.broadcast(dummy_sign, src = rec_src, group = send_groups[rec_src])
            sign_list[(pos + 1)%w_size].set_(dummy_sign)       
        #print('Step is : ', t)

        pos = (pos + 1)%w_size
        tensor_list[pos] = scale_list[pos] * sign_list[pos].float()
        #print(tensor_list[pos])
    final_tensor = torch.cat(tensor_list)
    origin_tensor.set_(final_tensor.mul_(1/w_size))
    buffer_error.set_((torch.cat(error_list)).mul_(1/w_size))
    #print(final_tensor[0:5])
    #if dist.get_rank() == 0 or dist.get_rank() == 5:
     #   print(tensor_list[0])
    #print('step 3 finished')
    #dist.barrier()
    return 0



def clean_sparse_simusend(Nankey, origin_tensor,his_tensor, origin_error, buffer_error, chunk_size, idx):
    if Nankey == 1:
        nantensor = torch.tensor(float('nan')).to(origin_tensor.device)
        origin_tensor[0].set_(nantensor)
        return -1
    temp_tensor = origin_tensor + origin_error
    sparsed_tensor,diff = advanced_sparse_compress(temp_tensor,his_tensor,chunk_size = chunk_size, idx = idx)

    dist.all_reduce(sparsed_tensor)
    sparsed_tensor.mul_(1.0/dist.get_world_size())
    buffer_error.set_(diff)

    origin_tensor.set_(sparsed_tensor)

    




def sparse_simusend(origin_tensor, his_tensor, origin_error, buffer_error, send_groups, chunk_size, idx):
    
    w_size = dist.get_world_size()
    #if dist.get_rank() == 0:
     #   print('Total length is :  ',torch.numel(origin_tensor))
    origin_list = torch.chunk(origin_tensor,w_size)
    his_list = torch.chunk(his_tensor,w_size)
    
    buffer_tensor = origin_tensor.clone().detach()
    sign_list,diff =  imple_sparse_compress(buffer_tensor + origin_error, his_tensor, chunk_size = chunk_size, idx = idx)
    sign_list = list(torch.chunk(sign_list,chunk_size))
    buffer_error.set_(diff)
        
    tensor_list = list(torch.chunk(sign_list, w_size))

    ori_error_list = torch.chunk(origin_error, w_size)
    error_list = list(torch.chunk(buffer_error, w_size))
    
    
    pos = (w_size - 1 - dist.get_rank() + 1)%(w_size)

    for t in range(w_size - 1):           
        sign_list[pos] = tensor_list[pos] 

        send_src = dist.get_rank()
        rec_src = (dist.get_rank() - 1)%w_size
        
        
        if dist.get_rank()%2 == 0:
            dummy_sign = torch.zeros_like(sign_list[pos])
            dummy_sign.set_(sign_list[pos])
            dist.broadcast(dummy_sign, src = send_src, group = send_groups[send_src])
        else:
            dummy_sign = torch.zeros_like(sign_list[(pos + 1)%w_size])
            dist.broadcast(dummy_sign, src = rec_src, group = send_groups[rec_src])
            #if dist.get_rank() == 1 and t==0:
             #   print('receive from ',rec_src,' gets ',dummy_sign[0:10])
            sign_list[(pos + 1)%w_size].set_(dummy_sign)
        if dist.get_rank()%2 == 1:
            dummy_sign = torch.zeros_like(sign_list[pos])
            dummy_sign.set_(sign_list[pos])
            dist.broadcast(dummy_sign, src = send_src, group = send_groups[send_src])
        else:
            dummy_sign = torch.zeros_like(sign_list[(pos + 1)%w_size])
            dist.broadcast(dummy_sign, src = rec_src, group = send_groups[rec_src])
            sign_list[(pos + 1)%w_size].set_(dummy_sign)       
        

        pos = (pos + 1)%w_size
        tensor_list[pos] = sign_list[pos] + origin_list[pos]
        
    #pos = (pos + 1)%w_size
    sign_list[pos] = tensor_list[pos] + ori_error_list[pos]
    tensor_list[pos] = sign_list[pos].float()

    for t in range(w_size - 1):
        send_src = dist.get_rank()
        rec_src = (dist.get_rank() - 1)%w_size
                
        if dist.get_rank()%2 == 0:
            dummy_sign = torch.zeros_like(sign_list[pos])
            dummy_sign.set_(sign_list[pos])
            dist.broadcast(dummy_sign, src = send_src, group = send_groups[send_src])
        else:
            dummy_sign = torch.zeros_like(sign_list[(pos + 1)%w_size])
            dist.broadcast(dummy_sign, src = rec_src, group = send_groups[rec_src])
            sign_list[(pos + 1)%w_size].set_(dummy_sign)
        if dist.get_rank()%2 == 1:
            dummy_sign = torch.zeros_like(sign_list[pos])
            dummy_sign.set_(sign_list[pos])
            dist.broadcast(dummy_sign, src = send_src, group = send_groups[send_src])
        else:
            dummy_sign = torch.zeros_like(sign_list[(pos + 1)%w_size])
            dist.broadcast(dummy_sign, src = rec_src, group = send_groups[rec_src])
            sign_list[(pos + 1)%w_size].set_(dummy_sign)       
        #print('Step is : ', t)

        pos = (pos + 1)%w_size
        tensor_list[pos] =  sign_list[pos].float()
        #print(tensor_list[pos])
    final_tensor = torch.cat(tensor_list)
    origin_tensor.set_(final_tensor.mul_(1.0/w_size))


    return 0



def mygather(origin_tensor, ecbuffer, buffer_error, final_ecbuffer, final_buffer_error, groups_list, wait_key_list):

    w_size = dist.get_world_size()
    rank = int(dist.get_rank()/2)
    #pos = (dist.get_rank() + 1)%w_size

    rec_tensor = torch.zeros_like(torch.chunk(origin_tensor, w_size/2)[dist.get_rank()])
    rec_scale = torch.zeros_like(origin_tensor[0])
    rec_sign = torch.zeros_like(rec_tensor)

    ori_error_list = torch.chunk(ecbuffer, w_size/2)
    error_list = torch.chunk(buffer_error, w_size/2)

    tensor_list = torch.chunk(origin_tensor, w_size/2)
    
    
    for t in range(w_size/2 - 1):
        send_pos = send_scheduler_ring(rank, t)
        rec_src = rec_scheduler_ring(rank, t)
        
        scale,sign_tensor,diff = imple_naive_compress(tensor_list[send_pos] + ori_error_list[send_pos])
        error_list[send_pos].set_(diff)
        #if dist.get_rank() == 0:
         #   print('Step is : ', t, ' compression leads to  ',sign_list[0][0:10])
        #print('Scale is: ',scale_list[pos])
        #dist.broadcast(dummy_tensor, src = 0, async_op = True)

        
        
        if wait_flag == 0:
            dummy_scale = scale.clone()
            dist.broadcast(dummy_scale, src = rank, group = groups_list[rank][send_pos])
        else:
            dummy_scale = torch.zeros_like(rec_scale)
            dist.broadcast(dummy_scale, src = rec_src, group = groups_list[rank][rec_src])
            rec_scale.set_(dummy_scale)            
        if wait_flag == 1:
            dummy_scale = scale.clone()
            dist.broadcast(dummy_scale, src = rank, group = groups_list[rank][send_pos])
        else:
            dummy_scale = torch.zeros_like(rec_scale)
            dist.broadcast(dummy_scale, src = rec_src, group = groups_list[rank][rec_src])
            rec_scale.set_(dummy_scale)
            
        
        if wait_flag == 0:
            dummy_sign = sign.clone()
            dist.broadcast(dummy_sign, src = rank, group = groups_list[rank][send_pos])
        else:
            dummy_sign = torch.zeros_like(rec_sign)
            dist.broadcast(dummy_sign, src = rec_src, group = groups_list[rank][rec_src])
            rec_sign.set_(dummy_sign)            
        if wait_flag == 1:
            dummy_sign = sign.clone()
            dist.broadcast(dummy_sign, src = rank, group = groups_list[rank][send_pos])
        else:
            dummy_sign = torch.zeros_like(rec_sign)
            dist.broadcast(dummy_sign, src = rec_src, group = groups_list[rank][rec_src])
            rec_sign.set_(dummy_sign)

        rec_tensor += rec_sign * rec_scale

    scale,sign_tensor,diff = imple_naive_compress(rec_tensor + final_ecbuffer)
    final_buffer_error.set_(diff)

    for t in range(w_size):
        if rank == t:
            dummy_scale = scale.clone().detach()
            dist.broadcast(dummy_scale,src = rank)
            dummy_sign = sign_tensor.clone().detach()
            dist.broadcast(dummy_sign,src = rank)
        else:
            dummy_scale = torch.zeros_like(scale)
            dist.broadcase(dummy_scale,src = t)
            dummy_sign = torch.zeros_like(torch.chunk(origin_tensor, w_size)[t])
            dist.broadcast(dummy_sign, src = t)
            tensor_list[t].set_(dummy_scale * dummy_sign)


def send_scheduler_ring(rank, t):
    send_pos = (rank + t + 1)%dist.get_world_size()
    return send_pos

def rec_scheduler_ring(rank, t):
    rec_src = (rank - t - 1)%dist.get_world_size()
    return rec_src

def group_send(src,dest,tensor,group_list):
    sender = src * 2 + 1
    receiver = dest * 2
    dist.broadcast(tensor,src = sender,group = group_list[sender][receiver])

def group_receive(src,dest,tensor,group_list):
    sender = src * 2 + 1
    receiver = dest * 2
    dist.broadcast(tensor,src = sender,group = group_list[sender][receiver])


