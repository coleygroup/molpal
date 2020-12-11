import argparse
import csv
from operator import itemgetter
from pathlib import Path
import pickle
import sys
from typing import Dict, List, Set

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
import numpy as np
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

NBINS=50

def get_num_iters(data_dir: str) -> int:
    scores_csvs = [p_csv for p_csv in Path(data_dir).iterdir()
                   if 'iter' in p_csv.stem]
    return len(scores_csvs)

def read_scores(scores_csv: str) -> Dict:
    """read the scores contained in the file located at scores_csv"""
    scores = {}
    failures = {}
    with open(scores_csv) as fid:
        reader = csv.reader(fid)
        next(reader)
        for row in reader:
            try:
                scores[row[0]] = float(row[1])
            except:
                failures[row[0]] = None
    
    return scores, failures

def get_new_points_by_epoch(scores_csvs: List[str]) -> List[Dict]:
    """get the set of new points and associated scores acquired at each
    iteration in the list of scores_csvs that are already sorted by iteration"""
    all_points = dict()
    new_points_by_epoch = []
    for scores_csv in scores_csvs:
        scores, _ = read_scores(scores_csv)
        new_points = {smi: score for smi, score in scores.items()
                      if smi not in all_points}

        new_points_by_epoch.append(new_points)
        all_points.update(new_points)
    
    return new_points_by_epoch

def add_ellipses(ax, invert=False):
    kwargs = dict(fill=False, color='white' if invert else 'black', lw=1.)
    ax.add_patch(Ellipse(xy=(6.05, -6.0), width=2.9, height=1.2, **kwargs))
    ax.add_patch(Ellipse(xy=(16.05, 4.5), width=1.7, height=2.6, **kwargs))

def add_model_data(fig, gs, data_dir, i, model,
                   d_smi_idx, fps_embedded, zmin, zmax,
                   portrait, n_models):
    scores_csvs = [p_csv for p_csv in Path(data_dir).iterdir()
                   if 'iter' in p_csv.stem]
    scores_csvs = sorted(scores_csvs, key=lambda p: int(p.stem.split('_')[4]))

    new_pointss = get_new_points_by_epoch(scores_csvs)
    if portrait:
        MAX_ROW = len(new_pointss)
    else:
        MAX_ROW = n_models

    axs = []
    for j, new_points in enumerate(new_pointss):
        if portrait:
            row, col = j, i
        else:
            row, col = i, j
        ax = fig.add_subplot(gs[row, col])
        smis, scores = zip(*new_points.items())
        idxs = [d_smi_idx[smi] for smi in smis]

        ax.scatter(
            fps_embedded[idxs, 0], fps_embedded[idxs, 1], 
            marker='.', c=scores, s=2, cmap='plasma', vmin=zmin, vmax=zmax
        )
        add_ellipses(ax)

        if row==0:
            if portrait:
                ax.set_title(model)

        if row==MAX_ROW:
            if not portrait:
                ax.set_xlabel(j)
        
        if col==0:
            if portrait:
                ax.set_ylabel(row)
            else:
                ax.set_ylabel(model)
        
        ax.set_xticks([])
        ax.set_yticks([])

        axs.append(ax)

    return fig, axs

def si_fig(d_smi_score, d_smi_idx, fps_embedded, data_dirs, models, 
           portrait=True):
    zmin = -max(score for score in d_smi_score.values() if score < 0)
    zmax = -min(d_smi_score.values())
    zmin = round((zmin+zmax)/2)

    n_models = len(data_dirs)
    n_iters = get_num_iters(data_dirs[0])

    if portrait:
        fig = plt.figure(figsize=(10*1.15, 15), constrained_layout=True)
        gs = fig.add_gridspec(nrows=n_iters, ncols=n_models)
    else:
        fig = plt.figure(figsize=(15*1.15, 10), constrained_layout=True)
        gs = fig.add_gridspec(nrows=n_models, ncols=n_iters)

    axs = []
    for i, (parent_dir, model) in enumerate(zip(data_dirs, models)):
        fig, axs_ = add_model_data(fig, gs, parent_dir, i, model,
                                   d_smi_idx, fps_embedded, zmin, zmax,
                                   portrait, n_models)
        axs.extend(axs_)

    ticks = list(range(zmin, round(zmax)))

    colormap = ScalarMappable(cmap='plasma')
    colormap.set_clim(zmin, zmax)
    cbar = plt.colorbar(colormap, ax=axs, aspect=30, ticks=ticks)
    cbar.ax.set_title('Score')

    ticks[0] = f'≤{ticks[0]}'
    cbar.ax.set_yticklabels(ticks)

    if portrait:
        fig.text(0.01, 0.5, 'Iteration', ha='center', va='center', 
                 rotation='vertical',
                 fontsize=14, fontweight='bold',)
        fig.text(0.465, 0.9975, 'Model', ha='center', va='top',
                 fontsize=14, fontweight='bold',)
    else:
        fig.text(0.01, 0.5, 'Model', ha='center', va='center',
                 rotation='vertical',
                 fontsize=16, fontweight='bold')
        fig.text(0.48, 0.01, 'Iteration', ha='center', va='center', 
                 fontsize=16, fontweight='bold',)

    plt.savefig(f'umap_fig_si_{"portrait" if portrait else "landscape"}.pdf')
    plt.clf()

def add_top1k_panel(fig, gs, d_smi_score, d_smi_idx, fps_embedded):
    true_top_1k = dict(sorted(d_smi_score.items(), key=itemgetter(1))[:1000])
    true_top_1k_smis = set(true_top_1k.keys())
    top_1k_idxs = [d_smi_idx[smi] for smi in true_top_1k_smis] 
    top_1k_fps_embedded = fps_embedded[top_1k_idxs, :]

    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.scatter(top_1k_fps_embedded[:, 0], top_1k_fps_embedded[:, 1],
                c='grey', marker='.')
    add_ellipses(ax)

    return fig, ax

def add_density_panel(fig, gs, ax1, fps_embedded):
    ax2 = fig.add_subplot(gs[0:2, 2:])
    _, _, _, im = ax2.hist2d(
        x=fps_embedded[:, 0], y=fps_embedded[:, 1],
        bins=NBINS, cmap='Purples_r'
    )
    ax2_cbar = plt.colorbar(im, ax=(ax1, ax2), aspect=20)
    ax2_cbar.ax.set_title('Points')
    
    ax2.set_yticks([])

    add_ellipses(ax2, True)

    return fig, ax2

def add_model_row(fig, gs, parent_dir, row, iters, model,
                  d_smi_idx, fps_embedded, zmin, zmax):
    scores_csvs = [p_csv for p_csv in Path(parent_dir).iterdir()
                   if 'iter' in p_csv.stem]
    scores_csvs = sorted(scores_csvs, key=lambda p: int(p.stem.split('_')[4]))

    col = 0
    axs = []
    for j, new_points in enumerate(get_new_points_by_epoch(scores_csvs)):
        if j not in iters:
            continue

        ax = fig.add_subplot(gs[row, col])
        smis, scores = zip(*new_points.items())
        idxs = [d_smi_idx[smi] for smi in smis]

        ax.scatter(
            fps_embedded[idxs, 0], fps_embedded[idxs, 1], alpha=0.75,
            marker='.', c=scores, s=2, cmap='plasma', vmin=zmin, vmax=zmax
        )
        add_ellipses(ax)

        if row==4:
            ax.set_xlabel(j)
        if col==0:
            ax.set_ylabel(model)


        ax.set_xticks([])
        ax.set_yticks([])

        axs.append(ax)
        col+=1

    return fig, axs

def main_fig(d_smi_score, d_smi_idx, fps_embedded, data_dirs,
             models=None, iters=None):
    models = ['RF', 'NN', 'MPN'] or models
    iters = [0, 1, 3, 5] or iters[:4]

    zmax = -min(d_smi_score.values())
    zmin = -max(score for score in d_smi_score.values() if score < 0)
    zmin = round((zmin+zmax)/2)

    nrows = 2+len(data_dirs)
    ncols = 4
    fig = plt.figure(figsize=(2*ncols*1.15, 2*nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=4)

    fig, ax1 = add_top1k_panel(fig, gs, d_smi_score, d_smi_idx, fps_embedded)
    fig, ax2 = add_density_panel(fig, gs, ax1, fps_embedded)
    
    axs = []
    for i, (data_dir, model) in enumerate(zip(data_dirs, models)):
        fig, axs_ = add_model_row(fig, gs, data_dir, i+2, iters, model,
                                  d_smi_idx, fps_embedded, zmin, zmax)
        axs.extend(axs_)
    
    colormap = ScalarMappable(cmap='plasma')
    colormap.set_clim(zmin, zmax)

    ticks = list(range(zmin, round(zmax)))

    cbar = plt.colorbar(colormap, ax=axs, aspect=30, ticks=ticks)
    cbar.ax.set_title('Score')

    ticks[0] = f'≤{ticks[0]}'
    cbar.ax.set_yticklabels(ticks)

    fig.text(-0.03, 1.03, 'A', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='center', ha='right')
    fig.text(-0.0, 1.03, 'B', transform=ax2.transAxes,
             fontsize=16, fontweight='bold', va='center', ha='left')
    fig.text(-0.03, -0.075, 'C', transform=ax1.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='right')

    fig.text(0.475, 0.005, 'Iteration', ha='center', va='center', 
             fontweight='bold')

    plt.savefig(f'umap_fig_main_2.pdf')
    plt.clf()

parser = argparse.ArgumentParser()
parser.add_argument('--scores-dict-pkl',
                    help='the filepath of a pickle file containing the scores dictionary')
parser.add_argument('--smis-csv',
                    help='the filepath of a csv file containing the SMILES string each molecule in the library in the 0th column')
parser.add_argument('--fps-embedded-npy',
                    help='a .npy file containing the 2D embedded fingerprint of each molecule in the library. Must be in the same order as smis-csv')
parser.add_argument('--data-dirs', nargs='+',
                    help='the directories containing molpal output data')
parser.add_argument('--models', nargs='+',
                    help='the respective name of each model used in --data-dirs')
parser.add_argument('--iters', nargs=4, type=int, default=[0, 1, 3, 5],
                    help='the FOUR iterations of points to show in the main figure')
parser.add_argument('--si-fig', action='store_true', default=False,
                    help='whether to produce generate the SI fig instead of the main fig')
parser.add_argument('--landscape', action='store_true', default=False,
                    help='whether to produce a landscape SI figure')

if __name__ == "__main__":
    args = parser.parse_args()

    d_smi_score = pickle.load(open(args.scores_dict_pkl, 'rb'))

    with open(args.smis_csv, 'r') as fid:
        reader = csv.reader(fid); next(reader)
        smis = [row[0] for row in tqdm(reader)]
    d_smi_idx = {smi: i for i, smi in enumerate(smis)}

    fps_embedded = np.load(args.fps_embedded_npy)

    if not args.si_fig:
        main_fig(d_smi_score, d_smi_idx, fps_embedded,
                 args.data_dirs, args.models, args.iters)
    else:
        si_fig(d_smi_score, d_smi_idx, fps_embedded,
               args.data_dirs, args.models, not args.landscape)