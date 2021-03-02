import argparse
from collections import Counter
from itertools import chain
from math import sqrt
import os
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import ray
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.SimDivFilters import rdSimDivPickers
import seaborn as sns
from tqdm import tqdm
import umap

sns.set_theme(style='white')

import utils

def get_nearest_neighbors(fps, i: int, k: int) -> np.ndarray:
    X_d = 1. - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
    I = np.argpartition(X_d, k+1)[:k+1]

    return I[X_d[I] != 0]

@ray.remote
def get_nearest_neighbors_batch(idxs: Sequence[int], fps, threshold: float) -> np.ndarray:
    def nns(i):
        X_d = 1. - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        return np.where(X_d < threshold)[0]

    return [nns(i) for i in idxs]

def nearest_neighbors(fps, threshold: float, chunksize: int = 16) -> List:
    chunksize = int(ray.cluster_resources()['CPU'] * chunksize)
    
    idxs = range(len(fps))
    fps = ray.put(fps)

    refs = [
        get_nearest_neighbors_batch.remote(idxs_chunk, fps, threshold)
        for idxs_chunk in tqdm(utils.chunks(idxs, chunksize))
    ]
    nn_idxs_chunks = [ray.get(r) for r in tqdm(refs, unit='idxs chunk')]
    
    return list(chain(*nn_idxs_chunks))

@ray.remote
def partial_hist_1d(
        idxs: Sequence[int], fps: Sequence,
        bins: int, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the sum of 1D histograms of the pairwise input distances
    bewteen index i and indices i+1: for all indices in idxs

    Parameters
    ----------
    idxs : Sequence[int]
        the indices
    fps : Sequence
        a sequence of fingerprints
    bins : int
        the number of bins to generate
    k : Optional[int], default=None
        the nearest neighbors to count. If None, use all pairwise distances

    Returns
    -------
    np.ndarray
        the histogram
    """
    hists = []
    for i in idxs:
        if k:
            X_d = 1. - np.array(
                DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            )
            X_d = np.delete(X_d, i)
            I = np.argpartition(X_d, k)
            X_d = X_d[I[:k]]
        else:
            X_d = 1. - np.array(
                DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
            )

        H, _ = np.histogram(X_d, bins=bins, range=(0., 1.))
        hists.append(H)

    return sum(hists)

@ray.remote
def partial_hist_2d(
        idxs: Sequence[int], fps: Sequence, scores: np.ndarray,
        bins: int, range_: Sequence, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the sum of 1D histograms of the pairwise input distances
    bewteen index i and indices i+1: for all indices in idxs

    Parameters
    ----------
    idxs : Sequence[int]
        the indices
    fps : Sequence
        a sequence of fingerprints
    scores : np.ndarray
    bins : int
        the number of bins to generate
    k : Optional[int], default=None
        the nearest neighbors to count. If None, use all pairwise distances

    Returns
    -------
    np.ndarray
        the histogram
    """
    hists = []
    for i in idxs:
        if k:
            X_d = 1. - np.array(
                DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            )
            Y_d = np.abs(scores[i+1:] - scores[i])

            X_d = np.delete(X_d, i)
            I = np.argpartition(X_d, k)

            X_d = X_d[I[:k]]
            Y_d = Y_d[I[:k]]
        else:
            X_d = 1. - np.array(
                DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
            )
            Y_d = np.abs(scores[i+1:] - scores[i])

        H, _, _ = np.histogram2d(
            X_d, Y_d, bins=bins, range=range_
        )
        hists.append(H.T)

    return sum(hists)

def distance_hist(
        fps: Sequence, scores: np.ndarray, bins: int, range_: Sequence,
        k: Optional[int] = None, two_d: bool = False, log: bool = False
    ) -> np.ndarray:

    chunksize = int(sqrt(ray.cluster_resources()['CPU']) * 4)
    idxs_chunks = utils.chunks(range(len(fps)), chunksize)

    fps = ray.put(fps)
    scores = ray.put(scores)
    bins = ray.put(bins)
    range_ = ray.put(range_)

    if not two_d:
        refs = [
            partial_hist_1d.remote(idxs, fps, bins, k)
            for idxs in idxs_chunks
        ]
    else:
        refs = [
            partial_hist_2d.remote(idxs, fps, scores, bins, range_, k)
            for idxs in tqdm(idxs_chunks)
        ]
    hists = (ray.get(r) for r in tqdm(refs, unit='chunk'))

    H = sum(hists)
    if log:
        H = np.log10(H, where=H > 0)

    return H

def cluster_mols(smis, fps: Sequence,
                 threshold: float = 0.65) -> List[int]:
    lp = rdSimDivPickers.LeaderPicker()
    idxs = list(lp.LazyBitVectorPick(fps, len(fps), threshold))
    centroids = [fps[i] for i in idxs]

    cluster_ids = []
    for fp in fps:
        T_cs = DataStructs.BulkTanimotoSimilarity(fp, centroids)
        T_cs = np.array(T_cs)

        i = np.argmin(T_cs)
        cid = idxs[i]
        cluster_ids.append(cid)
    
    return cluster_ids

def reduce_fps(fps = Sequence, length: int = 2048,
               k: Optional[int] = None, min_dist: float = 0.1):
    k = k or 15
    reducer = umap.UMAP(
        n_neighbors=k, metric='jaccard',
        min_dist=min_dist, n_jobs=len(os.sched_getaffinity(0))
    )
    X = np.empty((len(fps), length))
    for i, fp in enumerate(fps):
        ConvertToNumpyArray(fp, X[i])

    return reducer.fit_transform(X)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-csv')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--name')
    parser.add_argument('-N', type=int,
                        help='the number of top candidates in the dataset to look at')
    parser.add_argument('-k', type=int,
                        help='the number of nearest neighbors to use')
    parser.add_argument('--mode', default='neighbors',
                        choices=('viz', 'hist', 'cluster',
                                 'umap', 'cluster+umap'))
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--two-d', action='store_true', default=False)
    parser.add_argument('-t', '--threshold', type=float, default=0.65)
    parser.add_argument('--min-dist', type=float, default=0.1)

    args = parser.parse_args()

    d_smi_score = utils.parse_csv(
        args.score_csv, smiles_col=args.smiles_col, score_col=args.score_col,
    )

    if args.N:
        smis_scores = sorted(d_smi_score.items(), key=lambda kv: -kv[1])
        smis_scores = smis_scores[:args.N]
    else:
        smis_scores = list(d_smi_score.items())

    smis, scores = zip(*smis_scores)
    fps = utils.smis_to_fps(smis, args.radius, args.length)

    fig_dir = Path(f'sar/{args.name}/{args.mode}')
    fig_dir.mkdir(exist_ok=True, parents=True)

    top = f'top{args.N}' if args.N else 'all'

    if args.mode == 'viz':
        smis_scores = sorted(d_smi_score.items(), key=lambda kv: -kv[1])
        smis, scores = zip(*smis_scores)
        fps = utils.smis_to_fps(smis, args.radius, args.length)

        idxs = [random.randint(0, args.N) for _ in range(5)]
        for j, idx in tqdm(enumerate(idxs)):
            nn_idxs = get_nearest_neighbors(fps, idx, args.k)
            
            smis_ = [smis[idx], *[smis[i] for i in nn_idxs]]
            mols = [Chem.MolFromSmiles(smi) for smi in smis_]
            scores_ = [str(scores[idx]), *[str(scores[i]) for i in nn_idxs]]
            print(idx, nn_idxs)

            plot = Draw.MolsToGridImage(mols, legends=scores_)
            plot.save(f'{fig_dir}/{top}_{args.k}NN_{j}.png')

    elif args.mode == 'hist':
        fig = plt.figure()
        ax = fig.add_subplot(111)

        scores = utils.normalize(np.array(scores))

        range_ = [(0., 1.), (0., scores.max() - scores.min())]
        H = distance_hist(
            fps, scores, args.bins, range_, args.k, args.two_d, args.log
        )
        if not args.two_d:
            bins = np.linspace(0., 1., args.bins)
            width = (bins[1] - bins[0])
            ax.bar(bins, H, align='edge', width=width)
            ax.set_xlabel('Tanimoto distance')
            ax.set_ylabel('Count')
        else:
            xedges = np.linspace(0., 1., args.bins)
            yedges = np.linspace(0., range_[1][1], args.bins)

            fig = plt.figure()
            X, Y = np.meshgrid(xedges, yedges)
            ax = fig.add_subplot(111)
            pcm = ax.pcolormesh(X, Y, H, shading='auto')
            if args.log:
                fig.colorbar(pcm, ax=ax, label='log(Count)')
            else:
                fig.colorbar(pcm, ax=ax, label='Count')
            ax.set_ylabel('Score Distance')
            ax.set_xlabel('Tanimoto Distance')

        if args.k is None:
            comment = 'all'
        elif args.k >= 1:
            comment = f'{args.k} nearest neighbors'
        else:
            comment = f'{args.k} nearest neighbor'
        top = f'top-{args.N}' if args.N else 'all'

        ax.set_title(f'{args.name} {top} pairwise distances ({comment})')

        plt.tight_layout()
        fig.savefig(f'{fig_dir}/N_{args.N or "all"}_k_{args.k or "all"}_{2 if args.two_d else 1}D.png')

    elif args.mode == 'cluster':
        cids = cluster_mols(smis, fps, args.threshold)
        d_cid_size = Counter(cids)

        sizes = list(d_cid_size.values())
        size_counts = Counter(sizes)
        bins = len(size_counts) * 2

        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.suptitle(f'Cluster sizes in {args.name} {top} (t={args.threshold})')

        axs[0].hist(sizes, bins=bins)
        axs[0].set_ylabel('Count')

        axs[1].hist(sizes, log=True, bins=bins)
        axs[1].set_xlabel('Cluster size')
        axs[1].set_ylabel('Count')
        
        plt.tight_layout()
        t = f'{args.threshold}'.lstrip('0.')
        fig.savefig(f'{fig_dir}/{top}_t{t}.png')

    elif args.mode == 'umap':
        U = reduce_fps(fps, args.length, args.k, args.min_dist)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if args.N:
            vmin, vmax = None, None
        else:
            vmin, vmax = scores[len(scores)//2], scores[0]
        z = np.array(scores)
        order = np.argsort(z)
        im = ax.scatter(
            U[order, 0], U[order, 1], c=z[order], cmap='plasma',
            alpha=0.7, marker='.', vmin=vmin, vmax=vmax, 
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_alpha(1)
        cbar.draw_all()

        ax.set_title(f'Embedded fingerprints of {args.name} {top}')

        plt.tight_layout()
        fig.savefig(f'{fig_dir}/{top}_{args.k}NN.png')

    elif args.mode == 'cluster+umap':
        cids = cluster_mols(smis, fps, args.threshold)
        U = reduce_fps(fps, args.length, args.k, args.min_dist)

        d_cid_size = Counter(cids)
        # # n_large_clusters = sum(1 for size in d_cid_size.values() if size > 5)
        clabels = list(d_cid_size.keys())
        cvals = sns.color_palette('Accent', len(clabels) + 1)
        d_cid_color = dict(zip(clabels, cvals))
        colors = [d_cid_color[cid]# if cid in d_cid_color else cvals[-1]
                  for cid in cids]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.scatter(U[:, 0], U[:, 1], c=colors, alpha=0.7, marker='.')

        ax.set_title(f'Clustered fingerprints of {args.name} {top}')

        plt.tight_layout()

        t = f'{args.threshold}'.lstrip('0.')
        fig.savefig(f'{fig_dir}/{top}_{args.k}NN_t{t}.png')

if __name__ == "__main__":
    main()