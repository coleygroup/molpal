from typing import List, Optional

import torch
from torch import nn

class Projector(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 4,
                 layers: Optional[List[int]] = None,
                 threshold: float = 0.5):
        super().__init__()
        
        if layers:
            layers = [in_dim, *layers, out_dim]
        else:
            layers = [in_dim, out_dim]
        
        ffn = [
            nn.Linear(layers[i], layers[i+1])
            for i in range(len(layers)-1)
        ]
        self.ffn = nn.Sequential(*ffn)
        self.act = nn.Sigmoid()

        # threshold = threshold or 0.5
        self.t = threshold
        # self.t = torch.autograd.Variable(torch.Tensor([threshold]))

    def forward(self, x):
        x = self.act(self.ffn(x))
        return (x > self.t).to(torch.int8)# * 1
