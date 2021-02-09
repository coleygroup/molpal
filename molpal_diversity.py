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
    ray.init(num_cpus=len(os.sched_getaffinity(0)))

import simpot
import utils

def parse_csv(score_csv: str,
              k: Optional[int] = None) -> Dict[str, Optional[float]]:
    with open(score_csv, 'r') as fid:
        reader = csv.reader(fid)
        n_rows = sum(1 for _ in reader); fid.seek(0)
        next(reader)
        k = k or n_rows

        data = {
            row[0]: -float(row[1]) if row[1] else None
            for row in islice(reader, k)
        }
    
    return data

def smi_to_fp(smi, radius: int = 2, length: int = 2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smi), radius, length, useChirality=True
    )

def partial_distance_hist_1d(i: int, fps: Sequence,
                             bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the histogram of all pairwise distances bewteen
    index i and indices i+1:

    Parameters
    ----------
    i : int
        the index
    fps : Sequence
        a sequence of fingerprints
    bins : int
        the number of bins to generate

    Returns
    -------
    np.ndarray
        the histogram
    """

    X_sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    X_d = 1. - X_sim

    H, _ = np.histogram(X_d, bins=bins, range=(0., 1.))

    return H

def distance_hist(d_smi_score: Dict[str, Optional[float]],
                  bins: int,
                  one_d: bool = True) -> Tuple[np.ndarray, np.ndarray]: 
    smis, scores = zip(*d_smi_score.items())
    fps = [smi_to_fp(smi) for smi in smis]

    ray.put(fps)
    ray.put(bins)

    if one_d:
        @ray.remote
        def partial_distance_hist_1d_chunks(idxs: Sequence[int]):
            hists = [partial_distance_hist_1d(i, fps, bins) for i in idxs]
            return sum(hists)
        
        chunksize = int(ray.cluster_resources()['CPU'] * 4)
        idxs_chunks = utils.chunks(range(len(fps)), chunksize)
        refs = list(map(partial_distance_hist_1d_chunks.remote, idxs_chunks))
        hists = [ray.get(r) for r in tqdm(refs, unit='chunk')]

    else:
        Y = np.array(scores)
        Y = Y[Y != np.array(None)].astype('float')
        Y = utils.normalize(Y)
        ray.put(Y)

        range_ = [[0., 1.], [0., 6.]]
        @ray.remote
        def partial_distance_hist_2d_chunks(idxs: Sequence[int]):
            hists = [
                simpot.distance_hist(i, fps, Y, bins, range_) for i in idxs
            ]
            return sum(hists)
        
        chunksize = int(ray.cluster_resources()['CPU'] * 4)
        idxs_chunks = utils.chunks(range(len(fps)), chunksize)
        refs = list(map(partial_distance_hist_2d_chunks.remote, idxs_chunks))
        hists = [ray.get(r) for r in tqdm(refs, unit='chunk')]

    return sum(hists)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores-csvs', nargs='+')
    parser.add_argument('--name')
    parser.add_argument('-k', type=int)
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--two-d', action='store_true', default=False)

    args = parser.parse_args()
    data_by_iter = [
        parse_csv(score_csv, args.k) for score_csv in args.scores_csvs
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.linspace(0., 1., args.bins)
    width = (bins[1] - bins[0])

    if len(data_by_iter) == 1:
        hist = distance_hist(data_by_iter[0], args.bins, True)
        # print(hist)
        ax.bar(bins, hist, align='edge', width=width)

        # fig.savefig(f'{args.name}_hist.png')
        # exit()

    else:
        data_by_iter = sorted(data_by_iter, key=lambda d: len(d))
        new_data_by_iter = [data_by_iter[0]]
        for i in range(1, len(data_by_iter)):
            current_data = set(data_by_iter[i].items())
            previous_data = set(data_by_iter[i-1].items())
            new_data = dict(current_data ^ previous_data)
            new_data_by_iter.append(new_data)

        for i, data in enumerate(new_data_by_iter):
            hist = distance_hist(data, args.bins, True)
            # print(hist)
            ax.bar(bins, hist, align='edge', width=width,
                   label=f'iter_{i}', alpha=0.7)

        plt.legend()
        # fig.savefig(f'images/{args.name}/iter_{i}.png')
    
    ax.set_title(args.name)
    ax.set_xlabel('Tanimoto distance')
    ax.set_ylabel('Count')
    fig.savefig(f'images/molpal/{args.name}_hist.pdf')
        # print(len(d))
        # print(list(d.items())[:10])

if __name__ == "__main__":
    main()