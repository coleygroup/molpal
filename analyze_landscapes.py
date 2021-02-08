import argparse
import os
from typing import Tuple
import warnings

from matplotlib import pyplot as plt
from matplotlib import figure
import numpy as np
import seaborn as sns
from sklearn import metrics
import torch

from landscapes import build_landscape
from projector import Projector
from utils import normalize

sns.set_theme(style='white')

def get_points(landscape: str, dimension: int,
               N: int) -> Tuple[np.ndarray, np.ndarray]:
    landscape = build_landscape(landscape, dimension)

    X = landscape.sample(N)
    Y = normalize(landscape(X))

    return X, Y

def calculate_distances(X: np.ndarray, Y: np.ndarray,
                        metric: str) -> Tuple[np.ndarray, np.ndarray]:
    n_cpu = len(os.sched_getaffinity(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_d = metrics.pairwise_distances(X, metric=metric, n_jobs=n_cpu)

    Y_d = np.empty((Y.shape[0], Y.shape[0]))
    for i in range(Y_d.shape[0]):
        Y_d[i] = np.abs(Y - Y[i])

    idxs = np.triu_indices(X_d.shape[0])
    return X_d[idxs], Y_d[idxs]

def plot_hist2d(X, Y, nbins: int, metric: str) -> figure.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, _, _, im = ax.hist2d(X, Y, bins=nbins)

    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel(f'{metric.capitalize()} Distance')
    ax.set_ylabel('Score Distance')

    return fig

    # return scipy.stats.pearsonr(X, Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--landscape', default='ackley')
    parser.add_argument('-d', '--dimension', type=int, default=2)
    parser.add_argument('-N', type=int, default=10)
    parser.add_argument('--metric', default='jaccard')
    parser.add_argument('--nbins', type=int, default=50)
    parser.add_argument('--numpy-seed', type=int)
    parser.add_argument('--out-dim', type=int, default=10)
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('--torch-seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    X, Y = get_points(args.landscape, args.dimension, args.N)
    projector = Projector(X.shape[1], args.out_dim, args.layers, args.threshold)
    X = projector(torch.Tensor(X)).numpy()
    X_dists, Y_dists = calculate_distances(X, Y, args.metric)
    
    fig = plot_hist2d(X_dists, Y_dists, args.nbins, args.metric)

    base = (f'images/{args.landscape}_{args.N}_{args.metric}'
            + f'{args.out_dim}_hist2d_{args.nbins}')
    fig.savefig(f'{base}.png')
    # print(X_dists)
    # print(Y_dists)
    
    # s = f'{args.function},{args.dimension},{args.metric},{args.N},Y,{r:0.3f}'
    # with open('stats.csv', 'a') as fid:
    #     print(s, file=fid)