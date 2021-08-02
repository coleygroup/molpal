import argparse
from collections import Counter
from itertools import chain
from math import sqrt
import os
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

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

# def nearest_neighbors(fps, fp, T: int) -> Tuple[np.ndarray, np.ndarray]:
#     X_d = 1. - np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))
#     # I = np.argpartition(X_d, k+1)[:k+1]
#     I = np.where(X_d < T)[0]
#     # I = I[X_d[I] != 0]  # remove self

#     return I, X_d[I]

def nearest_neighbors(
    i_or_fp: Union[int, DataStructs.ExplicitBitVect],
    fps: Sequence[DataStructs.ExplicitBitVect],
    k_or_T: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the nearest neighbors to the given fingerprint in the list of
    fingerprints

    Parameters
    ----------
    i_or_fp : Union[int, DataStructs.ExplicitBitVector]
        either an index representing the fingerprint or the specific fingerprint
    fps : Sequence[DataStructs.ExplicitBitVector]
        the fingerprints to search
    k_or_T : Union[int, float]
        if int, the number of nearest neighbors to find ("k") or if a float,
        the threshold ("T") below which to consider a fingerprint a neighbor

    Returns
    -------
    I : np.ndarray
        the indices of the nearest neighbors in fps
    X_d : np.ndarray
        the corresponding distances
    """
    if isinstance(i_or_fp, int):
        fp = fps[i_or_fp]
    else:
        fp = i_or_fp
    
    X_d = 1. - np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))
    
    if isinstance(k_or_T, int):
        k = k_or_T
        if isinstance(i_or_fp, int):
            k = k+1

        I = np.argpartition(X_d, k)[:k]
    else:
        T = k_or_T
        I = np.where(X_d < T)[0]

    return I, X_d[I]

@ray.remote
def nearest_neighbors_batch(
    idxs: Sequence[int], fps, k_or_T: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [nearest_neighbors(i, fps, k_or_T) for i in idxs]

def nearest_neighbors_all(
    fps, k_or_T: float, chunksize: int = 16
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Get the nearest neighbors of each fingerprint in fps

    Parameters
    ----------
    fps : Sequence[DataStructs.ExplicitBitVector]
        the fingerprints to search
    k_or_T : Union[int, float]
        if int, the number of nearest neighbors to find ("k") or if a float,
        the threshold ("T") below which to consider a fingerprint a neighbor
    chunksize : int, default=16
        the size into which to chunk each job

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        a list of tuples containing where for each index i, contains the indices
        of all neighbors for fp[i] and their corresponding distances
    """
    chunksize = int(ray.cluster_resources()['CPU'] * chunksize)
    
    idxs = range(len(fps))
    fps = ray.put(fps)

    refs = [
        nearest_neighbors_batch.remote(idxs_chunk, fps, k_or_T)
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
            Y_d = np.abs(scores - scores[i])

            # print(len(X_d), len(Y_d))
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
        k: Optional[int] = None, two_d: bool = False
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

    return H, np.log10(H, where=H > 0)

def compute_centroids(fps: Sequence, similarity: float = 0.35) -> Tuple:
    distance = 1. - similarity
    lp = rdSimDivPickers.LeaderPicker()
    idxs = list(lp.LazyBitVectorPick(fps, len(fps), distance))
    return idxs, [fps[i] for i in idxs]

def cluster_mols(fps: Sequence,
                 similarity: float = 0.35) -> List[int]:
    idxs, centroids = compute_centroids(fps, similarity)

    cluster_ids = []
    for fp in fps:
        T_cs = np.array(DataStructs.BulkTanimotoSimilarity(fp, centroids))
        # T_cs = np.array(T_cs)

        i = np.argmax(T_cs)
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
                                 'umap', 'cluster+umap', 'cluster+viz'))
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--two-d', action='store_true', default=False)
    parser.add_argument('--similarity', type=float, default=0.35)
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
            nn_idxs, dists = nearest_neighbors(idx, fps, args.k)
            
            smis_ = [smis[i] for i in nn_idxs]
            scores_ = [scores[i] for i in nn_idxs]

            smis_scores_dists = sorted(
                zip(smis_, scores_, dists), key=lambda ssd: ssd[2]
            )

            mols = [Chem.MolFromSmiles(smi) for smi in smis_]
            legends = [
                f'score={score} | dist={dist:0.2f}'
                for _, score, dist in smis_scores_dists
            ]

            plot = Draw.MolsToGridImage(mols, legends=legends)
            plot.save(f'{fig_dir}/{top}_{args.k}NN_{j}.png')

        exit()

    elif args.mode == 'hist':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # scores = utils.normalize(np.array(scores))
        scores = np.array(scores)

        range_ = [(0., 1.), (0., scores.max() - scores.min())]
        H, logH = distance_hist(
            fps, scores, args.bins, range_, args.k, args.two_d
        )
        if not args.two_d:
            bins = np.linspace(0., 1., args.bins)
            width = (bins[1] - bins[0])
            ax1.bar(bins, H, align='edge', width=width)
            ax1.set_xlabel('Tanimoto distance')
            ax1.set_ylabel('Count')
            ax2.bar(bins, logH, align='edge', width=width)
            ax2.set_xlabel('Tanimoto distance')
            ax2.set_ylabel('log(Count)')
        else:
            xedges = np.linspace(0., 1., args.bins)
            yedges = np.linspace(0., range_[1][1], args.bins)
            X, Y = np.meshgrid(xedges, yedges)

            pcm1 = ax1.pcolormesh(X, Y, H, shading='auto')
            fig.colorbar(pcm1, ax=ax1, label='Count')
            pcm2 = ax2.pcolormesh(X, Y, logH, shading='auto')
            fig.colorbar(pcm2, ax=ax2, label='log(Count)')
            for ax in (ax1, ax2):    
                ax.set_ylabel('Score Distance')
                ax.set_xlabel('Tanimoto Distance')

        if args.k is None:
            comment = 'all'
        elif args.k >= 1:
            comment = f'{args.k} nearest neighbors'
        else:
            comment = f'{args.k} nearest neighbor'
        top = f'top-{args.N}' if args.N else 'all'

        fig.suptitle(f'{args.name} {top} pairwise distances ({comment})')

        plt.tight_layout()
        fig.savefig(
            f'{fig_dir}/N_{args.N or "all"}_k_{args.k or "all"}_{2 if args.two_d else 1}D.png'
        )

        exit()

    elif args.mode == 'cluster':
        cids = cluster_mols(fps, args.similarity)
        d_cid_size = Counter(cids)

        sizes = list(d_cid_size.values())
        size_counts = Counter(sizes)
        bins = np.arange(1, max(sizes)+1)

        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.suptitle(f'Cluster sizes in {args.name} {top} (s={args.similarity})')

        axs[0].hist(sizes, bins=bins)

        axs[1].hist(sizes, log=True, bins=bins)
        axs[1].set_xlabel('Cluster size')
        
        for ax in axs:
            ax.set_ylabel('Count')
            ax.minorticks_on()
            ax.set_xticks(np.arange(0, max(sizes)+5, 5))
            ax.grid(True, which='both', axis='x',)

        plt.tight_layout()
        t = f'{args.similarity}'.lstrip('0.')
        fig.savefig(f'{fig_dir}/{top}_t{t}.png')

        print(sum(sizes))

        name = input('Figure name: ')
        fig.savefig(f'figures/poster/{name}.pdf')

        exit()

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

        exit()

    elif args.mode == 'cluster+umap':
        cids = cluster_mols(fps, args.similarity)
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

        t = f'{args.similarity}'.lstrip('0.')
        fig.savefig(f'{fig_dir}/{top}_{args.k}NN_t{t}.pdf')
    
        exit()

    elif args.mode == 'cluster+viz':
        cids = cluster_mols(fps, args.similarity)
        d_cid_size = Counter(cids)

        smis_all, scores_all = list(zip(*d_smi_score.items()))
        fps_all = utils.smis_to_fps(smis_all, args.radius, args.length)

        SIZE = 1
        j = 0
        sizes = []
        for cid, smi in zip(cids, smis):
            if d_cid_size[cid] > SIZE:
                continue
            fp = utils.smi_to_fp(smi, args.radius, args.length)
            nn_idxs, dists_ = nearest_neighbors(fp, fps_all, 0.4)

            smis_ = [smis_all[i] for i in nn_idxs]
            scores_ = [scores_all[i] for i in nn_idxs]

            smis_scores_dists = zip(smis_, scores_, dists_)
            smis_scores_dists = sorted(smis_scores_dists,
                                       key=lambda ssd: ssd[2])
            
            # print(smis_scores_dists)
            mols_ = [Chem.MolFromSmiles(smi) for smi, _, _ in smis_scores_dists]
            legends = [
                f'score={score} | dist={dist:0.3f}'
                for _, score, dist in smis_scores_dists
            ]
            # print(legends)
            plot = Draw.MolsToGridImage(mols_, legends=legends)

            t = f'{args.similarity}'.lstrip('0.')
            plot.save(f'{fig_dir}/s{t}_minsize{SIZE}_{j}.pdf')
            
            j += 1
            sizes.append(len(mols_))
        
        print(Counter(sizes))
        exit()

    else:
        print('No mode selected!')
if __name__ == "__main__":
    main()