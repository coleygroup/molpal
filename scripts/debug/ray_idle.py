""" Simple script to reproduce issue with ray leaving idle processes during batch prediction """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # only because someone else using GPU 0 

import ray 
import torch
import subprocess
import numpy as np 
from tqdm import tqdm 
from itertools import islice

class SimpleModel: 
    """ a 'model' which takes an input vector with dimension dim 
    and outputs a single value """
    def __init__(self, dim, device='cuda'): 
        self.dim = dim 
        self.device = device
        self.model = torch.rand(dim)
        
    def update(self):
        random_inds = torch.randint(high=self.dim, size=(int(self.dim/10),))
        for ind in random_inds: 
            self.model[ind] = torch.rand(1)
        return self.model

    def to(self, device): 
        self.model = self.model.to(device)
        
    def predict(self, inputs): 
        model = ray.put(self.model)
        refs = [
            _predict.remote(
                model=model, inputs=torch.stack(list(inp)), device=self.device,
            ) for inp in batches(inputs, 1000)
        ]
        preds_chunks = [
            ray.get(r) for r in tqdm(refs, 'Prediction', unit='chunk', leave=False)
        ]
        preds_chunks = [
            ten.cpu() for ten in preds_chunks
        ]
        return np.concatenate(preds_chunks)
    
@ray.remote(num_gpus=1, max_calls=1)
def _predict(model, inputs, device):
    model = model.to(device)
    inputs = inputs.to(device)
    return torch.matmul(inputs, model)
    
def features(n_points: int = 100000, dim: int = 1000):
    feats = torch.rand(n_points, dim)
    return feats 

def batches(it, size: int):
    """Batch an iterable into batches of the given size, with the final batch
    potentially being smaller"""
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])

if __name__ == '__main__': 
    subprocess.call('nvidia-smi', shell=True)
    ray.init(num_cpus=40, num_gpus=1)
    mod = SimpleModel(dim=50, device='cuda')
    inputs = features(n_points=10000, dim=50)
    for i in range(0, 10): 
        mod.update()
        mod.predict(inputs)
        subprocess.call('nvidia-smi', shell=True)

