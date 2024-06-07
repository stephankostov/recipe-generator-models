# reference: https://github.com/dhlee347/pytorchic-bert

""" Utils Functions """

import yaml
import os
import random
import logging
import contextlib
import io
import sys
from box import Box

import numpy as np
import torch

def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return Box(config)

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

@contextlib.contextmanager
def nostdout():
    save_stdout = io.StringIO()
    sys.stdout = save_stdout
    yield
    sys.stdout = sys.__stdout__

def debugger_is_active():
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

def data_to_device(data, device):
    """
    Recursively move tensors to the specified device.
    Args:
        data: A tensor or a collection of tensors (nested tuples, lists, dicts).
        device: The target device (e.g., 'cuda:0').
    Returns:
        The same data structure with all tensors moved to the specified device.
    """
    if isinstance(data, (list, tuple)):
        return type(data)(data_to_device(x, device) for x in data)
    elif isinstance(data, dict):
        return {k: data_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data