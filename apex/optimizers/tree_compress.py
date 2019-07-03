from .FP16_compression import *
import torch


def tree_compress(t_list,error_list,inner_count=2):
    pass

def ring_compress(tensor_list,error_list):
    total_num = len(tensor_list)
    final_tensor = torch.zeros_like(tensor_list[0])
    for i in range(total_num):
        temp_tensor = naive_compress(final_tensor + tensor_list[i] + error_list[i])
        error_list[i] = final_tensor + tensor_list[i] + error_list[i] - temp_tensor
        final_tensor = temp_tensor
    final_tensor /= total_num
    return final_tensor
