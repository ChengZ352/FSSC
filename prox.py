import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sign_binary(x):
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    return torch.where(x >= 0, ones, zeros)

def prox(v, *, alpha_, lambda1_):
    """
    v has shape (m,) or (m, batches)
    u has shape (k,) or (k, batches)

    supports GPU tensors
    """
    #torch.Size([100, 784]) torch.Size([10, 784])
    norm_v = torch.norm(v, p=2, dim=0)
    #norm_u = torch.norm(u, p=2, dim=0)
    a = lambda1_ * alpha_/norm_v
    norm_v = sign_binary(1-a)
    
    if(norm_v.sum()<norm_v.shape[0]):
        #print(norm_v.sum(), norm_v.shape[0])
        a = a.detach().cpu().numpy()
        idx1 = np.argsort(a)[::-1][:(a>1).sum()]
        #pdb.set_trace()
    else:
        idx1 = np.array([])
    #return norm_v * v#, norm_u * u
    return norm_v * v, idx1


def inplace_prox(beta,  alpha_, lambda1_):
    beta.weight.data, idx1 = prox(
        beta.weight.data, alpha_=alpha_, lambda1_ = lambda1_
    )
    return idx1


def inplace_mask(beta,  mask):
    beta.weight.data  = beta.weight.data * mask