import torch
import numpy as np


class Permute():
    def __init__(self, n_trans):
        self.n_trans = n_trans
        self.name = 'permute'
    def apply(self, x):
        return permute_random(x, self.n_trans)

def permute_random(x, n_permutations=1):
    H, W = x.shape[-2], x.shape[-1]
    
    permuted_tensors = []
    
    for _ in range(n_permutations):
        perm_rows = torch.randperm(H)
        perm_cols = torch.randperm(W)
        
        x_permuted = x[..., perm_rows, :]
        x_permuted = x_permuted[..., :, perm_cols]
        
        permuted_tensors.append(x_permuted)
    
    x = torch.cat(permuted_tensors, dim=0)
    return x