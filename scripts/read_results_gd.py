from pathlib import Path 
from typing import List, Optional, Union
from molpal import args, pools, featurizer
from molpal.acquirer.pareto import Pareto
from molpal.objectives.lookup import LookupObjective
from configargparse import ArgumentParser
from plot_utils import set_style, method_colors, method_order

import matplotlib.pyplot as plt
import pygmo as pg 
from pymoo.indicators.gd import GD
import numpy as np 
import pickle 

set_style()

# paths_to_config = sorted((Path('moo_runs') / 'objective').glob('*min*'))
paths_to_config =  [Path('moo_runs/objective/JAK2_min.ini'), Path('moo_runs/objective/LCK_max.ini')]
# base_dir = Path('results/moo_results_2023-01-05-08-47')
base_dir = Path('results/moo_all_w_greedy_baseline')
figname = base_dir / 'dockstring_results_gd.png'
pool_sep = "\t"

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

def get_true_scores(paths_to_config: List):
    """
    for provided config file (assuming lookup objective), 
    finds true objectives and calculates hypervolume. 
    Assumes the smiles order is the same in all objective 
    csv files 
    """
    objs = [LookupObjective(str(path)) for path in paths_to_config]
    pool = get_pool(base_dir, pool_sep=pool_sep)
    data = [[obj.c*obj.data[smi] for smi in pool.smis()] for obj in objs]
    true_scores = np.array(data).T
    return objs, true_scores

def parse_config(config: Path): 
    parser = ArgumentParser()
    parser.add_argument('config', is_config_file=True)
    parser.add_argument('--metric', required=True)
    parser.add_argument('--conf-method')
    parser.add_argument('--cluster-type', default=None)
    parser.add_argument('--scalarize', default=False)
    
    args, unknown = parser.parse_known_args(str(config))

    return vars(args)

def calc_gd(scores, true_pf): 
    iter_pf = Pareto(num_objectives=true_scores.shape[1])
    iter_pf.update_front(scores)
    ind = GD(true_pf)
    return ind(iter_pf.front)

def get_expt_gds(run_dir: Path, true_scores, objs):
    iter_dirs = sorted((run_dir / 'chkpts').glob('iter_*'))
    gds = []

    pf = Pareto(num_objectives=true_scores.shape[1])
    pf.update_front(true_scores)
    true_pf = pf.front

    for iter in iter_dirs: 
        with open(iter/'scores.pkl', 'rb') as file: 
            explored = pickle.load(file)
        scores = np.array(list(explored.values()))
        if scores.ndim == 1: 
            acquired = list(explored.keys())
            scores = [list(obj(acquired).values()) for obj in objs]
            scores = np.array(scores).T
        
        gd = calc_gd(scores, true_pf)
        gds.append(gd)

    return np.array(gds)

def analyze_runs(base_dir: Path, true_scores, objs):
    folders = list(base_dir.rglob('iter_1'))
    folders = [folder.parent.parent for folder in folders]
    runs_gd = {}

    for folder in folders: 
        tags = parse_config(folder/'config.ini')
        expt_label = (tags['metric'], str(tags['cluster_type']), str(tags['scalarize']))
        gds = get_expt_gds(folder, true_scores, objs)

        if expt_label not in runs_gd.keys(): 
            runs_gd[expt_label] = gds[None,:]
        else: 
            runs_gd[expt_label] = np.concatenate((runs_gd[expt_label], gds[None,:]))

    return runs_gd 

def group_runs(runs): 
    expts = {}

    for label, hvs in runs.items(): 
        expts[label] = {}
        expts[label]['means'] = hvs.mean(axis=0)
        expts[label]['std'] = hvs.std(axis=0)
    
    return expts

def sort_expts(expts):
    sorted_expts = {}
    keys = sorted(list(expts.keys()))
    for key in keys: 
        sorted_expts[key] = expts[key]

    return sorted_expts 


def plot_by_cluster_type(expts, filename): 
    fig, axs = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8,2.8))
    axs[0].set_ylabel('Hypervolume fraction')
    axs[0].set_title('No Clustering')
    axs[1].set_title('Objective Clustering')
    axs[2].set_title('Feature Clustering')
    axs[3].set_title('Feature and Objective')
    for ax in axs: 
        ax.set_xlabel('Iteration')
    cls_to_ax = {'None':0, 'objs':1, 'fps':2, 'both': 3}

    for key, data in expts.items(): 
        if key[2] == 'True': 
            metric = f'scalar_{key[0]}'
        else: 
            metric = key[0]
        x = range(len(data['means']))
        ax = axs[cls_to_ax[key[1]]]
        ax.set_xticks(x)
        ax.plot(x, data['means'],color=method_colors[metric], label = metric)
        ax.fill_between(x, data['means'] - data['std'], data['means'] + data['std'], 
            facecolor=method_colors[metric], alpha=0.3
        )
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = plt.legend(by_label.values(), by_label.keys(), loc=7, frameon=False, facecolor="none", fancybox=False, )
    legend.get_frame().set_facecolor("none")
    
    fig.savefig(filename, bbox_inches="tight")
    
def plot_one_cluster_type(expts, filename):
    fig, axs = plt.subplots(1,1,figsize=(4,3))
    axs.set_ylabel('Generational Distance')
    axs.set_xlabel('Iteration')
    axs.set_title('JAK2/LCK Selectivity (MPN Surrogate)')    

    for key, data in expts.items():
        if key[2] == 'True': 
            metric = f'scalar_{key[0]}'
        else: 
            metric = key[0] 

        x = range(len(data['means']))
        axs.plot(x, data['means'],color=method_colors[metric], label = metric)
        axs.fill_between(x, data['means'] - data['std'], data['means'] + data['std'], 
            facecolor=method_colors[metric], alpha=0.3
        )

    legend = axs.legend(frameon=False, facecolor="none", fancybox=False, )
    legend.get_frame().set_facecolor("none")

    plt.savefig(filename)


if __name__ == '__main__': 
    objs, true_scores = get_true_scores(paths_to_config)
    runs_gd = analyze_runs(base_dir, true_scores, objs)
    expts = group_runs(runs_gd)
    expts = sort_expts(expts)
    plot_by_cluster_type(expts, figname)
    # plot_one_cluster_type(expts, figname)
    print('done')


     
