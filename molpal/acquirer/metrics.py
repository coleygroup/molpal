"""This module contains functions for calculating the acquisition score of an
input based on various metrics"""
from typing import Callable, Optional, Set, Iterable

import numpy as np
import pygmo as pg
import scipy.stats
from scipy.stats import norm
from molpal.acquirer.pareto import Pareto
import pygmo 
import ray
from tqdm import tqdm 

# this module maintains an independent random number generator
RG = np.random.default_rng()


def set_seed(seed: Optional[int] = None) -> None:
    """Set the seed of this module's random number generator"""
    global RG
    RG = np.random.default_rng(seed)


def get_metric(metric: str) -> Callable[..., float]:
    """Get the corresponding metric function"""
    try:
        return {
            'random': random,
            'threshold': random_threshold,
            'greedy': greedy,
            'noisy': noisy,
            'ucb': ucb,
            'lcb': lcb,
            'thompson': thompson,
            'ts': thompson,
            'ei': ei,
            'pi': pi,
            'nds': nds,
        }[metric]
    except KeyError:
        raise ValueError(f'Unrecognized metric: "{metric}"')


def get_needs(metric: str) -> Set[str]:
    """Get the values needed to compute this metric"""
    return {
        'random': set(),
        'greedy': {'means'},
        'noisy': {'means'},
        'ucb': {'means', 'vars'},
        'ei': {'means', 'vars'},
        'pi': {'means', 'vars'},
        'thompson': {'means', 'vars'},
        'ts': {'means', 'vars'},
        'threshold': {'means'},
        'nds': {'means'},
    }.get(metric, set())


def get_multiobjective(metric: str) -> bool:
    """Get whether metric is compatible with Pareto optimization"""
    return {
        'random': True,
        'greedy': False,
        'noisy': False,
        'ucb': False,
        'ei': True,
        'pi': True,
        'thompson': False,
        'ts': False,
        'threshold': False, 
        'nds': True,
    }.get(metric, set())


def calc(
    metric: str, Y_means: np.ndarray, Y_vars: np.ndarray,
    pareto_front: Pareto, current_max: float, threshold: float,
    beta: int, xi: float, stochastic: bool, nadir: np.ndarray,
    top_n_scored: int, 
) -> np.ndarray:
    """Call corresponding metric function with the proper args"""
    
    PF_points, _ = pareto_front.export_front()
    if metric == 'random':
        return random(Y_means)
    if metric == 'threshold':
        return random_threshold(Y_means, threshold)
    if metric == 'greedy':
        return greedy(Y_means, PF_points, nadir)
    if metric == 'noisy':
        return noisy(Y_means, PF_points, nadir)
    if metric == 'ucb':
        return ucb(Y_means, Y_vars, beta)
    if metric == 'lcb':
        return lcb(Y_means, Y_vars, beta)
    if metric in ('ts', 'thompson'):
        return thompson(Y_means, Y_vars, stochastic)
    if metric == 'ei':
        return ei(Y_means, Y_vars, current_max, xi, pareto_front=pareto_front)
    if metric == 'pi':
        return pi(Y_means, Y_vars, current_max, xi, pareto_front=pareto_front)
    if metric == 'nds':
        return nds(Y_means, top_n_scored)

    raise ValueError(f'Unrecognized metric: "{metric}"')


def random(Y_means: np.ndarray) -> np.ndarray:
    """Random acquistion score

    Parameters
    ----------
    Y_means : np.ndarray
        an array of length equal to the number of random scores to generate.
        It is only used to determine the dimension of the output array, so the
        values contained in it are meaningless.

    Returns
    -------
    np.ndarray
        the random acquisition scores
    """
    return RG.random(len(Y_means))


def random_threshold(Y_means: np.ndarray, threshold: float) -> float:
    """Random acquisition score [0, 1) if at or above threshold. Otherwise,
    return -1.

    Parameters
    ----------
    Y_means : np.ndarray
    threshold : float
        the threshold value below which to assign acquisition scores as -1 and
        above or equal to which to assign random acquisition scores in the
        range (0, 1]

    Returns
    -------
    np.ndarray
        the random threshold acquisition scores
    """
    return np.where(Y_means >= threshold, RG.random(Y_means.shape), -1.)


def greedy(
    Y_means: np.ndarray, PF_points: np.ndarray, nadir: np.ndarray
) -> np.ndarray:
    """Calculate the greedy acquisition utility of each point

    If the dimension of the objective is greather than 1, use the hypervolume
    improvement, or S-metric

    Parameters
    ----------
    Y_means : np.ndarray
        an NxT array where N is the number of inputs and T is dimension of the
        objective
    PF_points : np.ndarray
        points representing the current pareto frontier
    nadir : np.ndarray
        the nadir or reference point of the objective. I.e., the point in
        objective space which will be less than all other points in each
        dimension

    Returns
    -------
    np.ndarray
        the means if T is equal to 1 or the S-metric if T is greater than 1
        for each point
    """

    if Y_means.ndim == 1:
        return Y_means 
    
    if Y_means.shape[1] > 1:
        return hvi(Y_means, PF_points, nadir)

    return Y_means[:, 0]


def hvi(Y_means: np.ndarray,
        PF_points: np.ndarray,
        nadir: np.ndarray) -> np.ndarray:
    """Calculate the hypervolume improvement or S-metric for the predictions

    Parameters
    ----------
    Y_means : np.ndarray
    PF_points : np.ndarray
        points representing the current pareto frontier
    nadir : np.ndarray
        the nadir or reference point of the objective. I.e., the point in
        objective space which will be less than all other points in each
        dimension

    Returns
    -------
    S : np.ndarray
        the hypervolume improvement or S-metric for each point
    """
    Y_means = -Y_means
    P_f = -PF_points

    hv_old = pg.hypervolume(Y_means).compute(nadir)
    S = np.empty(len(Y_means))

    for i, Y_means in enumerate(Y_means):
        hv_new = pg.hypervolume(
            np.vstack((P_f, Y_means))
        ).compute(nadir)
        S[i] = hv_new - hv_old

    return S

@ray.remote
def hvpi(Y_means: np.array,
         Y_vars: np.array,
         pareto_front: Pareto) -> np.ndarray:
    """
    Calculate Hypervolume-based Probability of Improvement (HVPI).
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
    score : np.ndarray of size n_points with HVPI scores
    """
    Y_std = np.sqrt(Y_vars)
    N = Y_means.shape[0]
    n_obj = pareto_front.num_objectives

    # if pareto_front.reference_min is None:
    pareto_front.set_reference_min()
    reference_min = pareto_front.reference_min

    # Pareto front with reference points
    # shape: (front_size, n_obj)
    front = np.r_[
        np.array(reference_min).reshape((1, n_obj)),
        pareto_front.front,
        np.full((1, n_obj), np.inf),
    ]

    ax = np.arange(n_obj)
    n_cell = pareto_front.cells.lb.shape[0]

    # convert to minimization problem
    lower_bound = front[pareto_front.cells.ub, ax].reshape(
        (1, n_cell, n_obj)) * -1
    upper_bound = front[pareto_front.cells.lb, ax].reshape(
        (1, n_cell, n_obj)) * -1

    # convert to minimization problem
    Y_means = Y_means.reshape((N, 1, n_obj)) * -1
    Y_std = Y_std.reshape((N, 1, n_obj))

    # calculate cdf
    Phi_l = scipy.special.ndtr((lower_bound - Y_means) / Y_std)
    Phi_u = scipy.special.ndtr((upper_bound - Y_means) / Y_std)

    #  calculate PoI
    poi = np.sum(np.prod(Phi_u - Phi_l, axis=2), axis=1)  # shape: (N, 1)

    return poi

@ray.remote
def ehvi(Y_means: np.array,
         Y_vars: np.array,
         pareto_front: Pareto) -> np.ndarray:
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


def noisy(
    Y_means: np.ndarray, P_f: np.ndarray, nadir: np.ndarray
) -> np.ndarray:
    """Noisy greedy acquisition score

    Adds a random amount of noise to each predicted mean value. The noise is
    randomly sampled from a normal distribution centered at 0 with standard
    deviation equal to the standard deviation of the input predicted means.
    """
    sds = Y_means.std(axis=0)
    noise = RG.normal(scale=sds, size=Y_means.shape)
    return greedy(Y_means + noise, P_f, nadir)


def ucb(Y_means: np.ndarray, Y_vars: np.ndarray, beta: int = 2) -> float:
    """Upper confidence bound acquisition score

    Parameters
    ----------
    Y_means : np.ndarray
    Y_vars : np.ndarray
        the variance of the mean predicted y values
    beta : int (Default = 2)
        the number of standard deviations to add to Y_means

    Returns
    -------
    np.ndarray
        the upper confidence bound acquisition scores
    """
    return Y_means + beta*np.sqrt(Y_vars)


def lcb(Y_means: np.ndarray, Y_vars: np.ndarray, beta: int = 2) -> float:
    """Lower confidence bound acquisition score

    Parameters
    ----------
    Y_means : np.ndarray
    Y_vars : np.ndarray
    beta : int (Default = 2)

    Returns
    -------
    np.ndarray
        the lower confidence bound acquisition scores
    """
    return Y_means - beta*np.sqrt(Y_vars)


def thompson(Y_means: np.ndarray, Y_vars: np.ndarray,
             stochastic: bool = False) -> float:
    """Thompson acquisition score

    Parameters
    -----------
    Y_means : np.ndarray
    Y_vars : np.ndarray
    stochastic : bool
        is Y_means generated stochastically?

    Returns
    -------
    np.ndarray
        the thompson acquisition scores
    """
    if stochastic:
        return Y_means

    Y_sd = np.sqrt(Y_vars)

    return RG.normal(Y_means, Y_sd)


def ei(Y_means: np.ndarray, Y_vars: np.ndarray,
       current_max: float, xi: float = 0.01,
       pareto_front: Pareto = None) -> np.ndarray:
    """Exected improvement acquisition score

    Parameters
    ----------
    Y_means : np.ndarray
    Y_vars : np.ndarray
    current_max : float
        the current maximum observed score
    xi : float (Default = 0.01)
        the amount by which to shift the improvement score
    pareto : Pareto (Default = None)
        the Pareto front object corresponding to acquired points

    Returns
    -------
    E_imp : np.ndarray
        the expected improvement acquisition scores
    """
    if Y_means.ndim == 1 or Y_means.shape[1] == 1:
        improv = Y_means - current_max + xi
        Y_sd = np.sqrt(Y_vars)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = improv / Y_sd
        E_imp = improv * norm.cdf(Z) + Y_sd * norm.pdf(Z)

        # if the expected variance is 0, the expected improvement
        # is the predicted improvement
        mask = (Y_vars == 0)
        E_imp[mask] = improv[mask]

        return E_imp

    elif Y_means.shape[1] > 1:
        return chunked_ehvi(Y_means, Y_vars, pareto_front)


def pi(Y_means: np.ndarray, Y_vars: np.ndarray,
       current_max: float, xi: float = 0.01,
       pareto_front: Pareto = None) -> np.ndarray:
    """Probability of improvement acquisition score

    Parameters
    ----------
    Y_means : np.ndarray
    Y_vars : np.ndarray
    current_max : float
    xi : float (Default = 0.01)

    Returns
    -------
    P_imp : np.ndarray
        the probability of improvement acquisition scores
    """
    if Y_means.ndim == 1 or Y_means.shape[1] == 1:
        improv = Y_means - current_max + xi
        with np.errstate(divide='ignore'):
            Z = improv / np.sqrt(Y_vars)
        P_imp = norm.cdf(Z)

        # if expected variance is 0, probability of improvement is 0 or 1
        # depending on whether the predicted improvement is
        # negative or positive
        mask = (Y_vars == 0)
        P_imp[mask] = np.where(improv > 0, 1, 0)[mask]

        return P_imp
    elif Y_means.shape[1] > 1:
        return chunked_pi(Y_means, Y_vars, pareto_front)


def nds(Y_means: np.ndarray, top_n_scored: int):
    """
    Parameters:
    -----------
    Y_means: np.ndarray (number of points x number of objectives) 
    top_n_scored: int, the number of points to be assigned a rank. 
        Usually less than the total number of points, so 
        saves time by not computing rank for _all_ points

    Returns: 
    -------
    scores: np.ndarray (number of points,)
        Unranked points are assigned value of np.inf 
        scores are negative of ranks (lower ranks = better, less negative)

    """
    ranks = np.inf*np.ones([Y_means.shape[0]])
    Y_means_unranked = Y_means.copy()
    this_rank = 0

    # pbar = tqdm(total=top_n_scored)
    while np.isfinite(ranks).sum() < top_n_scored:
        this_rank = this_rank + 1
        if Y_means.shape[1] == 2: 
            front_num = pygmo.non_dominated_front_2d(-1*Y_means_unranked)
        else: 
            pf = Pareto(num_objectives=Y_means.shape[1], )
            pf.update_front(Y_means_unranked)
            front_num = pf.front_num

        ranks[front_num] = this_rank
        Y_means_unranked[front_num] = -np.inf
        # print(np.isfinite(ranks).sum())
        # pbar.n = np.isfinite(ranks).sum()
        # pbar.refresh()

    return -1*ranks

def chunked_ehvi(Y_means, Y_vars, pareto_front: Pareto, batch_size: int=1000): 
    
    n_chunks = np.ceil(Y_means.shape[0]/batch_size)
    Y_means_chunks = np.array_split(Y_means, n_chunks, axis=0)
    Y_vars_chunks = np.array_split(Y_vars, n_chunks, axis=0)
    
    stats = [
        ehvi.remote(
            Y_means_ch, Y_vars_ch, pareto_front
        ) 
        for Y_means_ch, Y_vars_ch in zip(Y_means_chunks, Y_vars_chunks)
    ]

    stats = [ray.get(stat) for stat in stats]
    
    return np.concatenate(stats)

def chunked_pi(Y_means, Y_vars, pareto_front: Pareto, batch_size: int=1000): 
    
    n_chunks = np.ceil(Y_means.shape[0]/batch_size)
    Y_means_chunks = np.array_split(Y_means, n_chunks, axis=0)
    Y_vars_chunks = np.array_split(Y_vars, n_chunks, axis=0)
    
    stats = [
        hvpi.remote(
            Y_means_ch, Y_vars_ch, pareto_front
        ) 
        for Y_means_ch, Y_vars_ch in zip(Y_means_chunks, Y_vars_chunks)
    ]

    stats = [ray.get(stat) for stat in stats]
    
    return np.concatenate(stats)


