"""This module contains functions for calculating the acquisition score of an
input based on various metrics"""
from typing import Callable, Optional, Set

import numpy as np
import pygmo as pg
from scipy.stats import norm

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
        'threshold': {'means'}
    }.get(metric, set())

def calc(
    metric: str, Y_means: np.ndarray, Y_vars: np.ndarray,
    P_f: np.ndarray, current_max: float, threshold: float,
    beta: int, xi: float, stochastic: bool, nadir: np.ndarray
) -> np.ndarray:
    """Call corresponding metric function with the proper args"""
    if metric == 'random':
        return random(Y_means)
    if metric == 'threshold':
        return random_threshold(Y_means, threshold)
    if metric == 'greedy':
        return greedy(Y_means, P_f, nadir)
    if metric == 'noisy':
        return noisy(Y_means, P_f, nadir)
    if metric == 'ucb':
        return ucb(Y_means, Y_vars, beta)
    if metric == 'lcb':
        return lcb(Y_means, Y_vars, beta)
    if metric in ('ts', 'thompson'):
        return thompson(Y_means, Y_vars, stochastic)
    if metric == 'ei':
        return ei(Y_means, Y_vars, current_max, xi)
    if metric == 'pi':
        return pi(Y_means, Y_vars, current_max, xi)

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
    Y_means: np.ndarray, P_f: np.ndarray, nadir: np.ndarray
) -> np.ndarray:
    """Calculate the greedy acquisition utility of each point
    
    If the dimension of the objective is greather than 1, use the hypervolume 
    improvement, or S-metric

    Parameters
    ----------
    Y_means : np.ndarray
        an NxT array where N is the number of inputs and T is dimension of the
        objective
    P_f : np.ndarray
        the current pareto frontier
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
    if Y_means.shape[1] > 1:
        return hvi(Y_means, P_f, nadir)

    return Y_means[:, 0]

def hvi(Y_means: np.ndarray, P_f: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    """Calculate the hypervolume improvement or S-metric for the predictions

    Parameters
    ----------
    Y_means : np.ndarray
    P_f : np.ndarray
        the current pareto frontier
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
    P_f = -P_f

    hv_old = pg.hypervolume(Y_means).compute(nadir)
    S = np.empty(len(Y_means))
    
    for i, Y_means in enumerate(Y_means):
        hv_new = pg.hypervolume(
            np.vstack((P_f, Y_means))
        ).compute(nadir)
        S[i] = hv_new - hv_old

    return S

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
       current_max: float, xi: float = 0.01) -> float:
    """Exected improvement acquisition score

    Parameters
    ----------
    Y_means : np.ndarray
    Y_vars : np.ndarray
    current_max : float
        the current maximum observed score
    xi : float (Default = 0.01)
        the amount by which to shift the improvement score

    Returns
    -------
    E_imp : np.ndarray
        the expected improvement acquisition scores
    """
    I = Y_means - current_max + xi
    Y_sd = np.sqrt(Y_vars)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = I / Y_sd
    E_imp = I * norm.cdf(Z) + Y_sd * norm.pdf(Z)

    # if the expected variance is 0, the expected improvement
    # is the predicted improvement
    mask = (Y_vars == 0)
    E_imp[mask] = I[mask]

    return E_imp

def pi(Y_means: np.ndarray, Y_vars: np.ndarray,
       current_max: float, xi: float = 0.01) -> np.ndarray:
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
    I = Y_means - current_max + xi
    with np.errstate(divide='ignore'):
        Z = I / np.sqrt(Y_vars)
    P_imp = norm.cdf(Z)

    # if expected variance is 0, probability of improvement is 0 or 1 
    # depending on whether the predicted improvement is negative or positive
    mask = (Y_vars == 0)
    P_imp[mask] = np.where(I > 0, 1, 0)[mask]

    return P_imp