"""This module contains functions for calculating the acquisition score of an
input based on various metrics"""
from typing import Callable, Optional, Set

import numpy as np
import numpy.random
from scipy.stats import norm

# this module maintains an independent random number generator
RG = np.random.default_rng()

def seed(seed: Optional[int] = None) -> None:
    global RG
    RG = np.random.default_rng(seed)

def get_metric(metric: str) -> Callable[..., float]:
    """Get the corresponding metric function"""
    try:
        return {
            'random': random_metric,
            'greedy': greedy,
            'noisy': noisy,
            'ucb': ucb,
            'ei': ei,
            'pi': pi,
            'thompson': thompson,
            'threshold': random_threshold
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
        'threshold': {'means'}
    }.get(metric, set())

def calc(metric: str, *args, **kwargs) -> np.ndarray:
    """Get the corresponding metric function and call it with the given args"""
    return get_metric(metric)(*args, **kwargs)

def random_metric(Y_mean: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Random acquistion score

    Parameters
    ----------
    Y_mean : np.ndarray
        an array of length equal to the number of random scores to generate.
        It is only used to determine the dimension of the output array
    
    Returns
    -------
    np.ndarray
        the random acquisition scores
    """
    return RG.random(len(Y_mean))

def greedy(Y_mean: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Greedy acquisition score
    
    Parameters
    ----------
    Y_mean : np.ndarray
        the mean predicted y values

    Returns
    -------
    np.ndarray
        the greedy acquisition scores
    """
    return Y_mean

def noisy(Y_mean: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Noisy acquisition score
    
    Adds a random amount of noise to each predicted mean value. The noise is
    randomly sampled from a normal distribution centered at 0 with standard
    deviation equal to the standard deviation of the input predicted means.
    """
    sd = np.std(Y_mean)
    noise = RG.normal(scale=sd, size=len(Y_mean))
    return Y_mean + noise

def ucb(Y_mean: np.ndarray, Y_var: np.ndarray, 
        beta: int = 2, **kwargs) -> float:
    """Upper confidence bound acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
        the variance of the mean predicted y values
    beta : int (Default = 2)
        the number of standard deviations to add to Y_mean

    Returns
    -------
    np.ndarray
        the upper confidence bound acquisition scores
    """
    return Y_mean + beta*np.sqrt(Y_var)

def lcb(Y_mean: np.ndarray, Y_var: np.ndarray, 
        beta: int = 2, **kwargs) -> float:
    """Lower confidence bound acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
    beta : int (Default = 2)

    Returns
    -------
    np.ndarray
        the lower confidence bound acquisition scores
    """
    return Y_mean - beta*np.sqrt(Y_var)

def ei(Y_mean: np.ndarray, Y_var: np.ndarray, current_max: float,
       xi: float = 0.01, **kwargs) -> float:
    """Exected improvement acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
    current_max : float
        the current maximum observed score
    xi : float (Default = 0.01)
        the amount by which to shift the improvement score

    Returns
    -------
    E_imp : np.ndarray
        the expected improvement acquisition scores
    """
    I = Y_mean - current_max + xi
    Y_sd = np.sqrt(Y_var)
    with np.errstate(divide='ignore'):
        Z = I / Y_sd
    E_imp = I * norm.cdf(Z) + Y_sd * norm.pdf(Z)

    # if the expected variance is 0, the expected improvement
    # is the predicted improvement
    mask = (Y_var == 0)
    E_imp[mask] = I[mask]

    return E_imp

def pi(Y_mean: np.ndarray, Y_var: np.ndarray, current_max: float,
       xi: float = 0.01, **kwargs) -> np.ndarray:
    """Probability of improvement acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
    current_max : float
    xi : float (Default = 0.01)

    Returns
    -------
    P_imp : np.ndarray
        the probability of improvement acquisition scores
    """
    I = Y_mean - current_max + xi
    with np.errstate(divide='ignore'):
        Z = I / np.sqrt(Y_var)
    P_imp = norm.cdf(Z)

    # if expected variance is 0, probability of improvement is 0 or 1 
    # depending on whether the predicted improvement is negative or positive
    mask = (Y_var == 0)
    P_imp[mask] = np.where(I > 0, 1, 0)[mask]

    return P_imp

def thompson(Y_mean: np.ndarray, Y_var: np.ndarray,
             stochastic: bool = False, **kwargs) -> float:
    """Thompson acquisition score

    Parameters
    -----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
    stochastic : bool
        is Y_mean generated stochastically?

    Returns
    -------
    np.ndarray
        the thompson acquisition scores
    """
    if stochastic:
        return Y_mean

    Y_sd = np.sqrt(Y_var)

    return RG.normal(Y_mean, Y_sd)

def random_threshold(Y_mean: np.ndarray, threshold: float) -> float:
    """Random acquisition score [0, 1) if above threshold. Otherwise,
    return -1.
    
    Parameters
    ----------
    Y_mean : np.ndarray
    threshold : float
        the threshold value below which to assign acquisition scores as -1 and 
        above or equal to which to assign random acquisition scores in the 
        range (0, 1]
    
    Returns
    -------
    np.ndarray
        the random threshold acquisition scores
    """
    return np.where(Y_mean >= threshold, RG.random(), -1.)
