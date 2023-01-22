""" Remote functions for multi-objective EI 
and PI calculations """

import ray 
import scipy 
import numpy as np
from molpal.acquirer.pareto import Pareto

@ray.remote 
def ndtr(x):
    return scipy.special.ndtr(x)

@ray.remote
def pdf(x):
    return scipy.stats.norm.pdf(x)

@ray.remote
def ehvi_remote(Y_means, Y_vars, pareto_front):
    """
    Calculate Expected Hyper-Volume Improvement (EHVI).
    Reference: (Couckuyt et al., 2014) Fast calculation of multiobjective
        probability of improvement and expected improvement criteria for
        Pareto optimization

    Parameters
    ----------
    Y_means : np.ndarray of size n_points x n_objs with prediction means
    Y_vars : np.ndarray of size n_points x n_objs with prediction variances
    pareto_front: a Pareto object storing information about the Pareto front

    Returns
    -------
    score : np.ndarray of size n_points with EHVI scores
    """
    Y_std = np.sqrt(Y_vars)
    N = Y_means.shape[0]
    n_obj = pareto_front.num_objectives

    # if pareto_front.reference_min is None:
    pareto_front.set_reference_min()
    # if pareto_front.reference_max is None:
    pareto_front.set_reference_max()
    reference_min = pareto_front.reference_min
    reference_max = pareto_front.reference_max

    # Pareto front with reference points
    # shape: (front_size, n_obj)
    front = np.r_[
        np.array(reference_min).reshape((1, n_obj)),
        pareto_front.front,
        np.array(reference_max).reshape((1, n_obj)),
    ]

    ax = np.arange(n_obj)

    # convert to minimization problem
    lower_bound = front[pareto_front.cells.ub, ax] * -1
    upper_bound = front[pareto_front.cells.lb, ax] * -1

    n_cell = pareto_front.cells.lb.shape[0]

    # shape: (n_cell, 1, n_cell, n_obj)
    lower_bound = np.tile(lower_bound, (n_cell, 1, 1, 1))
    upper_bound = np.tile(upper_bound, (n_cell, 1, 1, 1))
    a = lower_bound.transpose((2, 1, 0, 3))
    b = upper_bound.transpose((2, 1, 0, 3))

    # convert to minimization problem
    Y_means = Y_means.reshape((1, N, 1, n_obj)) * -1
    Y_std = Y_std.reshape((1, N, 1, n_obj))

    # calculate pdf, cdf
    phi_min_bu = scipy.stats.norm.pdf((np.minimum(b, upper_bound)
                                       - Y_means) / Y_std)
    phi_max_al = scipy.stats.norm.pdf((np.maximum(a, lower_bound)
                                       - Y_means) / Y_std)

    Phi_l = scipy.special.ndtr((lower_bound - Y_means) / Y_std)
    Phi_u = scipy.special.ndtr((upper_bound - Y_means) / Y_std)
    Phi_a = scipy.special.ndtr((a - Y_means) / Y_std)
    Phi_b = scipy.special.ndtr((b - Y_means) / Y_std)

    # calculate G
    is_type_A = np.logical_and(a < upper_bound, lower_bound < b)
    is_type_B = upper_bound <= a

    # note: Phi[max_or_min(x,y)] = max_or_min(Phi[x], Phi[y])
    EI_A = (
        (b - a) * (np.maximum(Phi_a, Phi_l) - Phi_l)
        + (b - Y_means) * (np.minimum(Phi_b, Phi_u) - np.maximum(Phi_a, Phi_l))
        + Y_std * (phi_min_bu - phi_max_al)
    )
    EI_B = (b - a) * (Phi_u - Phi_l)

    G = EI_A * is_type_A + EI_B * is_type_B
    score = np.sum(np.sum(np.prod(G, axis=3), axis=0), axis=1)  # shape: (N, 1)
    return score

def chunked_stat(x, func, batch_size: int = 100000): 
    x_new = x.reshape((x.size,))
    n_chunks = np.ceil(x.size/batch_size)
    x_chunks = np.array_split(x_new, n_chunks)
    stats = [func.remote(chunk) for chunk in x_chunks]
    stats = [ray.get(stat) for stat in stats]
    stats = np.concatenate(stats).reshape(x.shape)
    return stats

def chunked_ehvi(Y_means, Y_vars, pareto_front: Pareto, batch_size: int=1000): 
    
    n_chunks = np.ceil(Y_means.shape[0]/batch_size)
    Y_means_chunks = np.array_split(Y_means, n_chunks, axis=0)
    Y_vars_chunks = np.array_split(Y_vars, n_chunks, axis=0)
    
    stats = [
        ehvi_remote.remote(
            Y_means_ch, Y_vars_ch, pareto_front
        ) 
        for Y_means_ch, Y_vars_ch in zip(Y_means_chunks, Y_vars_chunks)
    ]

    stats = [ray.get(stat) for stat in stats]
    
    return np.concatenate(stats)


    



