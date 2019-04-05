import types
import torch
import importlib

class FusedAdamNew(torch.optim.Optimizer):