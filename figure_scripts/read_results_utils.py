from configargparse import ArgumentParser
from pathlib import Path 
from typing import List, Union, Dict
from molpal import args, pools, featurizer
from molpal.objectives.lookup import LookupObjective
import numpy as np 
import pygmo as pg 
import pickle 
from molpal.acquirer.pareto import Pareto
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def parse_config(config: Path): 
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('config', is_config_file=True)
    parser.add_argument('--metric', required=True)
    parser.add_argument('--conf-method')
    parser.add_argument('--cluster-type', default=None)
    parser.add_argument('--scalarize', default=False)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--model-seed', default=None)
    
    args, unknown = parser.parse_known_args(str(config))

    return vars(args)

def get_pool(base_dir: Union[Path, str], pool_sep = None): 
    """ Gets a pool from a config file. Assumes the pool in all runs in a base directory are the same """
    config_file = list(base_dir.rglob("config.ini"))[0]
    parser = ArgumentParser()
    args.add_pool_args(parser)
    args.add_general_args(parser)
    args.add_encoder_args(parser)
    params, unknown = parser.parse_known_args("--config " + str(config_file))
    params = vars(params)
    if pool_sep: 
        params['delimiter'] = pool_sep

    feat = featurizer.Featurizer(
        fingerprint=params['fingerprint'],
        radius=params['radius'], length=params['length']
    )
    pool = pools.pool(featurizer=feat, **params)
    return pool

def calc_hv(points: np.ndarray, ref_point = None):
    """ 
    Calculates the hypervolume assuming maximization of a list of points 
    points: np.ndarray of shape (n_points, n_objs) 
    """
    n_objs = points.shape[1]

    # change positive docking scores off-target to zero, as pos. scores add no value 
    points[:,1] = np.clip(points[:,1], a_min=None, a_max=0)
    points[:,0] = np.clip(points[:,0], a_min=0, a_max=None)

    if ref_point is None: 
        front_min = points.min(axis=0, keepdims=True)
        w = np.max(points, axis=0, keepdims=True) - front_min
        reference_min = front_min - w * 2 / 10
    else: 
        reference_min = ref_point
     
    # convert to minimization (how pygmo calculates hv)
    points = -points 
    ref_point = -reference_min
    
    # compute hypervolume
    hv = pg.hypervolume(points)
    hv = hv.compute(ref_point.squeeze())

    return hv, reference_min


def calc_true_hv(paths_to_config: List, base_dir, pool_sep=','):
    """
    for provided config file (assuming lookup objective), 
    finds true objectives and calculates hypervolume. 
    Assumes the smiles order is the same in all objective 
    csv files 
    """
    objs = [LookupObjective(str(path)) for path in paths_to_config]
    pool = get_pool(base_dir, pool_sep=pool_sep)
    smis = list(pool.smis())
    data = [[obj.c*obj.data[smi] for smi in smis] for obj in objs]
    data = np.array(data).T
    hv, reference_min = calc_hv(data.copy())
    return hv, reference_min, objs, data, smis 

def get_hvs(all_scores, reference_min, objs = None):  
    hvs = []
    for scores in all_scores:         
        hv, _ = calc_hv(scores.copy(), ref_point=reference_min)
        hvs.append(hv)
        
    return np.array(hvs)

def get_scores(run_dir: Path, objs: List[LookupObjective]):
    iter_dirs = sorted((run_dir / 'chkpts').glob('iter_*'))
    all_scores = []
    all_acquired = []
    explored_smis = []

    for iter in iter_dirs: 
        with open(iter/'scores.pkl', 'rb') as file: 
            explored = pickle.load(file)
        with open(iter/'new_scores.pkl', 'rb') as file: 
            acquired = pickle.load(file)

        scores = np.array(list(explored.values()))
        if scores.ndim == 1: 
            smis = list(explored.keys())
            scores = [list(obj(smis).values()) for obj in objs]
            scores = np.array(scores).T

        explored_smis.append(list(explored.keys()))
        all_scores.append(scores)

        acquired_scores = np.array(list(acquired.values()))
        if acquired_scores.ndim == 1: 
            smis = list(acquired.keys())
            acquired_scores = [list(obj(smis).values()) for obj in objs]
            acquired_scores = np.array(acquired_scores).T

        all_acquired.append(acquired_scores)

    return all_scores, all_acquired, explored_smis

def pareto_profile(all_scores, reference_min, true_pf): 
    pf = Pareto(num_objectives=2)
    pf.set_reference_min(reference_min)
    fronts = []
    
    for scores in all_scores: 
        pf.update_front(scores)
        front, _ = pf.export_front()
        front = np.unique(front, axis=0)
        fronts.append(front)

    final_front = front
    interp_front_x = np.linspace(np.min(true_pf[:,0]), np.max(true_pf[:,0]), 5*true_pf.shape[0])
    interp_y = np.interp(interp_front_x, true_pf[:,0], true_pf[:,1])
    interp_pf = np.vstack([interp_front_x, interp_y]).T
    pf = np.vstack([true_pf, interp_pf])

    ind = GD(pf)
    final_front_gd = ind(final_front)
    all_gd = ind(all_scores[-1])

    pf = true_pf.copy()
    pf[:,1] = np.clip(pf[:,1], a_min=None, a_max=0)
    pf[:,0] = np.clip(pf[:,0], a_min=0, a_max=None)

    ind = IGD(pf)
    final_igd = ind(final_front)
    all_igd = ind(all_scores[-1])

    return fronts, final_front, final_front_gd, all_gd, final_igd, all_igd 

def calc_front_extent(front): 
    from scipy.spatial.distance import pdist
    return np.max(pdist(front))


def true_pf(true_scores, reference_min): 
    pf = Pareto(num_objectives=2)
    pf.set_reference_min(reference_min)
    pf.update_front(true_scores)
    front, _ = pf.export_front()
    return front

def top_k_smiles(k, smiles: List[str], scores: np.array):
    
    scores_unranked = scores.copy()
    this_rank = 0
    top_smis = []
    smiles = np.array(smiles)

    # pbar = tqdm(total=top_n_scored)
    while len(top_smis)/len(smiles) < k:
        this_rank = this_rank + 1

        front_num = pg.non_dominated_front_2d(-1*scores_unranked)
        scores_unranked[front_num] = -np.inf
        top_smis.extend([smiles[i] for i in front_num.astype(int)])
    
    top_percent = len(top_smis)/len(smiles)
    
    return top_smis, top_percent

def get_top_k_profile(explored_smis, top_k_smis): 
    top_k_profile = []
    for smis in explored_smis: 
        intersect = set(smis) & set(top_k_smis)
        frac = len(intersect)/len(top_k_smis)
        top_k_profile.append(frac)
    
    return top_k_profile

def top_k_smis_from_pool(k, pool, paths_to_config):
    smis = list(pool.smis())
    objs = [LookupObjective(str(path)) for path in paths_to_config]
    data = [[obj.c*obj.data[smi] for smi in smis] for obj in objs]
    scores = np.array(data).T
    top_smis, _ = top_k_smiles(k, smis, scores)
    return top_smis