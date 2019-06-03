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

def naive_compress(gradient):
    l2 = gradient.norm()
    sign_tensor = gradient.sign()
    sign_tensor_l2 = sign_tensor.norm()
    scale = l2 / sign_tensor_l2
    grad = scale * sign_tensor


    return gradient

def tenary_compress(grad):
    thres = grad.abs().max()
    scale_grad = grad.abs()/thres + 0.01
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

    gradient_tensor = grad.flatten() * mask.half()
    gradient_tensor = gradient_tensor.view(sp)

    return gradient_tensor


