import torch
import random
import numpy as np


class Rotate():
    def __init__(self, n_trans, max_offset=0):
        self.n_trans = n_trans
        self.max_offset=max_offset
        self.name = 'rotation'
    def apply(self, x):
        return rotate_random(x, self.n_trans)

def rotate_random(x, n_rotations=5):
    rotated_tensors = []
    
    for _ in range(n_rotations):
        angle = random.randint(1,3)
        
        rotated_tensor = x.clone()
        rotated_tensor = torch.rot90(rotated_tensor, k=int(angle / 90), dims=(-2, -1))
        
        rotated_tensors.append(rotated_tensor)
    
    x = torch.cat(rotated_tensors, dim=0)
    
    return x