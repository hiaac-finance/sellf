import numpy as np
import torch

def graph_detach(*args):
    return [arg.detach() for arg in args]


def to_dytype_device(dtype, device, *args):
    return [arg.to(dtype).to(device) for arg in args]

def to_device(device, *args):
    return [arg.to(device) for arg in args]

def torch_to_numpy(*args):
    if len(args) == 1:
        return args[0].detach().cpu().numpy()
    else:
        return [arg.detach().cpu().numpy() for arg in args]
    
class RunningStats:
    """
    Calculate normalized input from running mean and std
    See https://www.johndcook.com/blog/standard_deviation/
    """
    def __init__(self, clip=1e6):
        self.x = 0 # Current value of data stream
        self.mean = 0 # Current mean
        self.sumsq = 0 # Current sum of squares, used in var/std calculation

        self.var = 0 # Current variance
        self.std = 0 # Current std

        self.count = 0 # Counter

        self.clip = clip

    def push(self, x):
        self.x = x
        self.count += 1
        if self.count == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean += (x - self.mean) / self.count
            self.sumsq += (x - old_mean) * (x - self.mean)
            self.var = self.sumsq / (self.count - 1)
            self.std = np.sqrt(self.var)
            self.std = np.maximum(self.std, 1e-2)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def get_std(self):
        return self.std

    def normalize(self, x=None):
        if x is not None:
            self.push(x)
            if self.count <= 1:
                return self.x
            else:
                output= (self.x - self.mean) / self.std
        else:
            output = (self.x - self.mean) / self.std
        return np.clip(output, -self.clip, self.clip)