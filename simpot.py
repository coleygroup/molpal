import argparse
import math
import csv
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ray
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
import scipy.stats
import seaborn as sns
from tqdm import trange

from utils import Timer, pmap, normalize

if not ray.is_initialized():
    # ray.init(num_cpus=len(os.sched_getaffinity(0)))
    ray.init(address='auto')

sns.set_theme(style='white')

def parse_scores_csv(scores_csv,
                     title_line: bool = True,
                     smiles_col: int = 0,
                     score_col: int = 1) -> List[Tuple[str, float]]:
    smis_scores = []
    with open(scores_csv, 'r') as fid:
        if title_line:
            reader = next(fid)
        reader = csv.reader(fid)

        for row in reader:
            try:
                smi, score = row[smiles_col], float(row[score_col])
                smis_scores.append((smi, score))
            except (ValueError, IndexError):
                continue

    return smis_scores

def max_dist(i: int, X: np.ndarray):
    return max(np.abs(X[i+1:] - X[i]))

def smi_to_fp(smi, radius: int = 2, length: int = 2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smi), radius, length, useChirality=True
    )

def distances(
    i: int, fps: Sequence, Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    X_sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    X_d = 1. - X_sim
    Y_d = np.abs(Y[i+1:] - Y[i])

    return X_d, Y_d

def distance_hist(
    i: int, fps: Sequence, Y: np.ndarray, bins: int, range: List[List]
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the 2D histogram of all pairwise input and output distances 
    bewteen index i and indices i+1:

    Parameters
    ----------
    i : int
        the index
    fps : Sequence
        a sequence of fingerprints (input points)
    Y : np.ndarray
        a sequence of scores (output points)
    bins : int
        the number of x and y bins to generate
    range : List[List]
        a 2D list of the form [[xmin, xmax], [ymin, max]] from which to
        construct the histogram

    Returns
    -------
    np.ndarray
        the transposed 2D histogram
    """

    X_sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    X_d = 1. - X_sim
    Y_d = np.abs(Y[i+1:] - Y[i])

    H, _, _ = np.histogram2d(X_d, Y_d, bins=bins, range=range)

    return H.T

def partial_distance_avgs(
    i: int, fps: Sequence, Y: np.ndarray
) -> Tuple[float, float, int]:

    X_sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    X_d = 1. - X_sim
    Y_d = np.abs(Y[i+1:] - Y[i])

    return X_d.mean(), Y_d.mean(), len(X_d)

def avg(xs: Sequence[float], ws: Sequence[float]):
    """
    calculate the overall sequence average from the weighted averages

    Parameters
    ----------
    xs : Sequence[float]
        the weighted averages
    ws : Sequence[float]
        their respective weights

    Returns
    -------
    float
        the overall average
    """
    X = np.array(xs)
    W = np.array(ws)

    return (X*W).sum() / W.sum()

def cov_dev(
    i: int, fps: Sequence, Y: np.ndarray, x_bar: float, y_bar: float
) -> Tuple[float, float, float]:
    """calculate the sum of covariance formula terms for for all pairwise
    comparisons to index i

    Parameters
    ----------
    i : int
        the index
    fps : Sequence
        a sequence of fingerprints from which to generate the pairwise
        distance vector of fingerprints
    Y : np.ndarray
        a parallel vector of scores from which to generate the pairwise
        distance vector of scores
    x_bar : float
        the average fingerprint distance
    y_bar : float
        the average score distance

    Returns
    -------
    float
        the sum of cross-terms: SUM[(x_i - xbar)*(y_i - ybar)]
    float
        the sum of x-terms: SUM[(x_i - xbar)^2]
    float
        the sum of y-terms: SUM[(y_i - ybar)^2]
    """

    X_sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    X_d = 1. - X_sim
    Y_d = np.abs(Y[i+1:] - Y[i])

    X_dev = X_d - x_bar
    Y_dev = Y_d - y_bar

    return (X_dev*Y_dev).sum(), (X_dev*X_dev).sum(), (Y_dev*Y_dev).sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores-csv')
    parser.add_argument('--smiles-col', type=int)
    parser.add_argument('--score-col', type=int)
    parser.add_argument('--radius', type=int)
    parser.add_argument('--length', type=int)
    parser.add_argument('-N', type=int)
    parser.add_argument('--nbins', type=int, default=50)
    args = parser.parse_args()

    data = Path(args.scores_csv).stem

    smis_scores = parse_scores_csv(
        args.scores_csv, smiles_col=args.smiles_col, score_col=args.score_col
    )
    if args.N and len(smis_scores) > args.N:
        smis_scores = random.sample(smis_scores, args.N)
    smis, scores = zip(*smis_scores)

    chunksize = int(ray.cluster_resources()['CPU'] * 512)
    fps = pmap(smi_to_fp, smis, chunksize, args.radius, args.length)[:args.N]
    Y = np.array(scores)[:args.N]
    Y = normalize(Y)
    N = min(len(Y), len(fps))

    #--------------------------------------------------------------------------#

    # chunksize = int(ray.cluster_resources()['CPU'] * 4)
    # avgs = pmap(partial_distance_avgs, range(len(Y)-1), chunksize,
    #             fps=fps, Y=Y)
    # x_avgs, y_avgs, ws = zip(*avgs)
    # x_bar = avg(x_avgs, ws)
    # y_bar = avg(y_avgs, ws)

    # covs_devs = pmap(cov_dev, range(len(Y)-1), chunksize,
    #                  fps=fps, Y=Y, x_bar=x_bar, y_bar=y_bar)
    # # print(covs_devs[:5])
    # covs, sq_x_devs, sq_y_devs = zip(*covs_devs)

    # a = sum(c for c in covs)
    # b = math.sqrt(sum(x for x in sq_x_devs))
    # c = math.sqrt(sum(y for y in sq_y_devs))

    # r = a / (b*c)

    #--------------------------------------------------------------------------#

    xmin, xmax = 0., 1.
    ymin = 0.

    chunksize = int(ray.cluster_resources()['CPU'] * 32)
    ymax = max(pmap(max_dist, range(len(Y)-1), chunksize, X=Y))
    hist_range = [[xmin, xmax], [ymin, ymax]]

    chunksize = int(ray.cluster_resources()['CPU'] * 4)
    H_partials = pmap(
        distance_hist, range(len(Y)-1), chunksize,
        fps=fps, Y=Y, bins=args.nbins, range=hist_range
    )
    # H = np.zeros((args.nbins, args.nbins))
    # for H_partial in H_partials:
    #     H += H_partial
    H = sum(H_partials)
    
    xedges = np.linspace(xmin, xmax, args.nbins)
    yedges = np.linspace(ymin, ymax, args.nbins)

    fig = plt.figure()
    X, Y = np.meshgrid(xedges, yedges)
    ax = fig.add_subplot(111)
    pcm = ax.pcolormesh(X, Y, H, shading='auto')
    fig.colorbar(pcm, ax=ax, label='Count')
    ax.set_ylabel('Score Distance')
    ax.set_xlabel('Tanimoto Distance')

    base = f'images/{data}_{N}_distance_hist2d_{args.nbins}'
    # fig.savefig(f'{base}.pdf')
    fig.savefig(f'{base}.png')

    # r, _ = scipy.stats.pearsonr(X_d, Y_d)

    # data = Path(args.scores_csv).stem
    # s = f'{data},{args.length},tanimoto,{N},Y,{r:0.3f}'
    # with open('stats.csv', 'a') as fid:
    #     print(s, file=fid)

if __name__ == '__main__':
    main()