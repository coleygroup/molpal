""" Remote functions for multi-objective EI 
and PI calculations """

import ray 
import scipy 
import numpy as np

@ray.remote 
def ndtr(x):
    return scipy.special.ndtr(x)

@ray.remote
def pdf(x):
    return scipy.stats.norm.pdf(x)

def chunked_stat(x, func, batch_size: int = 100000): 
    x_new = x.reshape((x.size,))
    n_chunks = np.ceil(x.size/batch_size)
    x_chunks = np.array_split(x_new, n_chunks)
    stats = [func.remote(chunk) for chunk in x_chunks]
    stats = [ray.get(stat) for stat in stats]
    stats = np.concatenate(stats).reshape(x.shape)
    return stats

