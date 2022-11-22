import torch

def print_cuda():
    t = torch.cuda.get_device_properties(0).total_memory / 10 ** 6
    r = torch.cuda.memory_reserved(0) / 10 ** 6
    a = torch.cuda.memory_allocated(0) / 10 ** 6
    c = torch.cuda.memory_cached(0) / 10 ** 6
    f = t - r - a 
    print("total {} reserved {} allocated {} free {} cached {}".format(t, r, a, f, c))