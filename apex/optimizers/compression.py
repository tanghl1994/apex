import torch
import torch.nn as nn
import torch.nn.functional as F


import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse


from importlib import reload

def imple_naive_compress(gradient):
    l2 = gradient.norm()
    sign_tensor = gradient.sign()
    sign_tensor_l2 = sign_tensor.norm()
    scale = l2 / sign_tensor_l2
    diff = gradient - scale * sign_tensor
    #sign_tensor = gradient / scale
    sign_tensor = (sign_tensor + 1)/2
    
    return scale,sign_tensor,diff

def test_naive_compress(gradient):
    l2 = gradient.norm()
   # sign_tensor = gradient.sign()
   # sign_tensor_l2 = sign_tensor.norm()
    scale = torch.zeros_like(gradient[0]) + 1.0
    diff = torch.zeros_like(gradient)
    sign_tensor = gradient.clone().detach()
    sign_tensor = (sign_tensor + 1.0)/2.0
    
    return scale,sign_tensor,diff


def imple_random_sparse_compress(gradient, alpha = 0.999):
   
    random_tensor = torch.rand_like(gradient)
    mask_tensor = random_tensor < alpha
    sign_tensor = gradient * mask_tensor.float()
    
    diff = gradient.data - sign_tensor.data
    #sign_tensor = gradient.clone().detach()
    #sign_tensor = (sign_tensor + 1.0)/2.0
    
    return sign_tensor,diff


def imple_sparse_compress(gradient,alpha = 0.9):
   
    scale = torch.zeros_like(gradient[0]) + 1.0
    buffer_tensor = torch.chunk(gradient,chunk_size)
    sign_tensor = torch.chunk(torch.zeros_like(gradient), chunk_size)
    sign_tensor[idx].set_(buffer_tensor[idx])
    sign_tensor = torch.cat(sign_tensor)
    
    diff = gradient.data - sign_tensor.data
    #sign_tensor = gradient.clone().detach()
    #sign_tensor = (sign_tensor + 1.0)/2.0
    
    return sign_tensor,diff





def naive_compress(gradient):
    l2 = gradient.norm()
    sign_tensor = gradient.sign()
    sign_tensor_l2 = sign_tensor.norm()
    scale = l2 / sign_tensor_l2
    gradient = scale * sign_tensor


    return gradient

def tenary_compress(grad):
    thres = grad.abs().max()
    scale_grad = grad.abs()/thres + 0.05
    cut_tensor = torch.zeros_like(scale_grad) + 1.0
    scale_grad = torch.min(scale_grad,cut_tensor)
    random_tensor = torch.rand_like(scale_grad)
    mask_tensor = (scale_grad>=random_tensor).float()

    gradient = grad * mask_tensor
    gradient = gradient / scale_grad

    return gradient


def topk_compress(grad,alpha = 0.03):
    num = int((grad.nelement()*alpha)) + 1
    sp = grad.shape

    absgradient_tensor = grad.flatten().abs()
    mask = absgradient_tensor >= absgradient_tensor.topk(k = num)[0][-1]

    gradient_tensor = grad.flatten()
    gradient_tensor = gradient_tensor.view(sp)

    return gradient_tensor


