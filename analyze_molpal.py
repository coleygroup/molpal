import argparse
from collections import Counter, defaultdict
import csv
from itertools import chain, islice, takewhile
import math
from operator import itemgetter
import os
from pathlib import Path
import pickle
import pprint
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ray
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
import seaborn as sns
from tqdm import tqdm, trange

sns.set_theme(style='white')

if not ray.is_initialized():
    try:
        ray.init('auto')
    except ConnectionError:
        ray.init(num_cpus=len(os.sched_getaffinity(0)))

# import simpot
import utils

@ray.remote
def smi_to_fp_chunk(smis, radius: int = 2, length: int = 2048) -> List:
    return [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), radius, length, useChirality=True
        ) for smi in smis
    ]

@ray.remote
def partial_distance_hist_1d(
    idxs: Sequence[int], fps: Sequence, bins: int, N: Optional[int] = None
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
    N : Optional[int], default=None
        the nearest neighbors to count. If None, use all pairwise distances

    Returns
    -------
    np.ndarray
        the histogram
    """
    hists = []
    for i in idxs:
        if N:
            X_d = 1. - np.array(
                DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            )
            X_d = np.delete(X_d, i)
            I = np.argpartition(X_d, N)
            X_d = X_d[I[:N]]
        else:
            X_d = 1. - np.array(
                DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
            )

        H, _ = np.histogram(X_d, bins=bins, range=(0., 1.))
        hists.append(H)

    return sum(hists)

@ray.remote
def partial_distance_hist_2d(
    idxs: Sequence[int], fps: Sequence, Y: np.ndarray,
    bins: int, range_: List[List]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the sum of 2D histograms of the pairwise (input, output) distances
    bewteen index i and indices i+1: for all indices in idxs

    Parameters
    ----------
    idxs : Sequence[int]
        the indices
    fps : Sequence
        a sequence of fingerprints
    Y : np.ndarray
        a sequence of scores (output points)
    bins : int
        the number of x and y bins to generate
    range_ : List[List]
        a 2D list of the form [[xmin, xmax], [ymin, max]] from which to
        construct the histogram

    Returns
    -------
    np.ndarray
        the histogram
    """
    hists = []
    for i in idxs:
        X_sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
        X_d = 1. - X_sim
        Y_d = np.abs(Y[i+1:] - Y[i])

        H, _, _ = np.histogram2d(X_d, Y_d, bins=bins, range=range_)

        hists.append(H.T)

    return sum(hists)

def distance_hist(d_smi_score: Dict[str, Optional[float]],
                  bins: int, N: Optional[int] = None,
                  one_d: bool = True) -> Tuple[np.ndarray, np.ndarray]: 
    smis, scores = zip(*d_smi_score.items())

    chunksize = int(ray.cluster_resources()['CPU'] * 512)
    smis_chunks = utils.chunks(smis, chunksize)
    refs = [smi_to_fp_chunk.remote(smis) for smis in smis_chunks]
    fps_chunks = [ray.get(r) for r in tqdm(refs)]
    fps = list(chain(*fps_chunks))

    ray.put(fps)
    ray.put(bins)

    if one_d:
        chunksize = int(ray.cluster_resources()['CPU'] * 4)
        idxs_chunks = utils.chunks(range(len(fps)), chunksize)
        refs = [
            partial_distance_hist_1d.remote(idxs, fps, bins, N)
            for idxs in idxs_chunks
        ]
        hists = [ray.get(r) for r in tqdm(refs, unit='chunk')]

    else:
        Y = np.array(scores)
        Y = Y[Y != np.array(None)].astype('float')
        Y = utils.normalize(Y)
        ray.put(Y)

        range_ = [[0., 1.], [0., 6.]]
        # @ray.remote
        # def partial_distance_hist_2d_chunks(idxs: Sequence[int]):
        #     hists = [
        #         simpot.distance_hist(i, fps, Y, bins, range_) for i in idxs
        #     ]
        #     return sum(hists)
        
        chunksize = int(ray.cluster_resources()['CPU'] * 4)
        idxs_chunks = utils.chunks(range(len(fps)), chunksize)
        refs = [
            partial_distance_hist_2d.remote(idxs, fps, Y, bins, range_)
            for idxs in idxs_chunks
        ]
        hists = [ray.get(r) for r in tqdm(refs, unit='chunk')]

    return sum(hists)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores-csvs', nargs='+')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('--name')
    parser.add_argument('-k', type=int)
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('-N', type=int)
    parser.add_argument('--two-d', action='store_true', default=False)

    args = parser.parse_args()
    data_by_iter = [
        utils.parse_csv(
            score_csv, smiles_col=args.smiles_col,
            score_col=args.score_col, k=args.k
        ) for score_csv in args.scores_csvs
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.linspace(0., 1., args.bins)
    width = (bins[1] - bins[0])

    if len(data_by_iter) == 1:
        sub_dir = 'sar'
        # print(len(data_by_iter[0]))

        hist = distance_hist(data_by_iter[0], args.bins, args.N, True)
        # print(hist)
        ax.bar(bins, hist, align='edge', width=width)

        # fig.savefig(f'{args.name}_hist.png')
        # exit()

    else:
        sub_dir = 'molpal'

        data_by_iter = sorted(data_by_iter, key=lambda d: len(d))
        new_data_by_iter = [data_by_iter[0]]
        for i in range(1, len(data_by_iter)):
            current_data = set(data_by_iter[i].items())
            previous_data = set(data_by_iter[i-1].items())
            new_data = dict(current_data ^ previous_data)
            new_data_by_iter.append(new_data)

        for i, data in enumerate(new_data_by_iter):
            hist = distance_hist(data, args.bins, True)
            ax.bar(bins, hist, align='edge', width=width,
                   label=f'iter_{i}', alpha=0.7)

        plt.legend()
        # fig.savefig(f'images/{args.name}/iter_{i}.png')
    neighbors = f'nearest{args.N}' if args.N else 'all'

    ax.set_title(f'{args.name} top-{args.k} {neighbors}')
    ax.set_xlabel('Tanimoto distance')
    ax.set_ylabel('Count')

    plt.tight_layout()
    fig_dir = Path(f'images/{sub_dir}/{args.name}')
    fig_dir.mkdir(exist_ok=True, parents=True)
    hist_type = '2d' if args.two_d else '1d'
    fig.savefig(f'{fig_dir}/top{args.k}_{neighbors}_{hist_type}.pdf')

if __name__ == "__main__":
    main()