import argparse
from collections import Counter
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ray
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.SimDivFilters import rdSimDivPickers
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white')

import utils

def get_nearest_neighbors(fps, i: int, k: int) -> np.ndarray:
    X_d = 1. - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
    I = np.argpartition(X_d, k+1)[:k+1]

    return I[X_d[I] != 0]

@ray.remote
def partial_hist(
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

def distance_hist(
        fps: Sequence, bins: int,
        k: Optional[int] = None, log: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]: 
    ray.put(fps)
    ray.put(bins)

    chunksize = int(ray.cluster_resources()['CPU'] * 4)
    idxs_chunks = utils.chunks(range(len(fps)), chunksize)
    refs = [
        partial_hist.remote(idxs, fps, bins, k)
        for idxs in idxs_chunks
    ]
    hists = [ray.get(r) for r in tqdm(refs, unit='chunk')]

    H = sum(hists)
    if log:
        H = np.log10(H)

    return H

def cluster_mols(smis, fps: Sequence,
                 threshold: float = 0.65) -> Dict[int, int]:
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
    
    return Counter(cluster_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-csv')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('--name')
    parser.add_argument('-N', type=int,
                        help='the number of top candidates in the dataset to look at')
    parser.add_argument('-k', type=int,
                        help='the number of nearest neighbors to use')
    parser.add_argument('--mode', default='neighbors',
                        choices=('viz', 'hist', 'cluster'))
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('-t', '--threshold', type=float)

    args = parser.parse_args()
    d_smi_score = utils.parse_csv(
        args.score_csv, smiles_col=args.smiles_col,
        score_col=args.score_col,
    )
    if args.N:
        smis_scores = sorted(d_smi_score.items(), key=lambda kv: -kv[1])
        smis_scores = smis_scores[:args.N]

    smis, scores = zip(*smis_scores)
    fps = utils.smis_to_fps(smis)

    fig_dir = Path(f'sar/{args.name}/{args.mode}')
    fig_dir.mkdir(exist_ok=True, parents=True)

    top = f'top{args.N}' if args.N else 'all'

    if args.mode == 'viz':
        idxs = [random.randint(0, len(fps)) for _ in range(5)]
        for j, idx in enumerate(idxs):
            nn_idxs = get_nearest_neighbors(fps, idx, args.k)
            
            smis_ = [smis[idx], *[smis[i] for i in nn_idxs]]
            mols = [Chem.MolFromSmiles(smi) for smi in smis_]
            scores_ = [str(scores[idx]), *[str(scores[i]) for i in nn_idxs]]
            print(idx, nn_idxs)

            plot = Draw.MolsToGridImage(mols, legends=scores_)
            plot.save(f'{fig_dir}/{top}_{args.k}NN_{j}.png')

    elif args.mode == 'hist':
        bins = np.linspace(0., 1., args.bins)
        width = (bins[1] - bins[0])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        hist = distance_hist(fps, args.bins, args.k, args.log)
        ax.bar(bins, hist, align='edge', width=width)

        neighbors = f'{args.k}NN' if args.k else 'all'
        ax.set_title(f'{args.name} {neighbors} distances')
        ax.set_xlabel('Tanimoto distance')
        ax.set_ylabel('Count')

        plt.tight_layout()
        fig.savefig(f'{fig_dir}/{top}_{neighbors}.pdf')

    elif args.mode == 'cluster':
        d_cid_size = cluster_mols(smis, fps, args.threshold)
        sizes = list(d_cid_size.values())
        size_counts = Counter(sizes)
        bins = len(size_counts) * 2
        print(sorted(sizes, reverse=True))

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

if __name__ == "__main__":
    main()