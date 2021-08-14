from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from operator import itemgetter
from pathlib import Path
import pickle
import pprint

from experiment import Experiment
from typing import List, Set, Sequence, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ExplicitBitVect
from rdkit.SimDivFilters import rdSimDivPickers
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

def smi_to_fp(smi: str, radius: int, length: int):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smi), radius, length, useChirality=True
    )

def compute_centroids(
    fps: Sequence[ExplicitBitVect], similarity: float = 0.35
) -> Tuple[List[int], List[ExplicitBitVect]]:
    """Compute the cluster centers of a list of molecular fingerprints based
    on sphere clustering

    Parameters
    ----------
    fps : Sequence[ExplicitBitVect]
        the molecular fingerprints from which to identify the cluster centers
    similarity : float, default=0.35
        the maximum allowable similarity between cluster centroids

    Returns
    -------
    idxs : List[int]
        the indices of the centroids in the input list of fingerprints
    """
    distance = 1. - similarity
    lp = rdSimDivPickers.LeaderPicker()
    idxs = list(lp.LazyBitVectorPick(fps, len(fps), distance))
    return idxs

def assign_cluster_label(
    fp: ExplicitBitVect,
    centroids: Sequence[ExplicitBitVect], labels: Sequence[int]
) -> int:
    """assign the cluster label to the input fingerprint

    Parameters
    ----------
    fp : ExplicitBitVect
        the fingerprint to assign a label to
    centroids : Sequence[ExplicitBitVect]
        the cluster centroids
    labels : Sequence[int]
        their labels

    Returns
    -------
    int
        the label of the closest cluster centroid
    """
    i = np.array(
        DataStructs.BulkTanimotoSimilarity(fp, centroids)
    ).argmax()
    return labels[i]
    # return labels[np.array(
    #     DataStructs.BulkTanimotoSimilarity(fp, centroids)
    # ).argmax()]

def cluster_fps(
    fps: Sequence[ExplicitBitVect], similarity: float = 0.35
) -> List[int]:
    """Cluster the input molecular fingerprints and return their associated 
    labels

    Parameters
    ----------
    fps : Sequence[ExplicitBitVect]
        the fingerprints to cluster
    similarity : float, default=0.35
        the maximum allowable similarity between cluster centers

    Returns
    -------
    List[int]
        the cluster label of each fingerprint
    """
    idxs = compute_centroids(fps, similarity)
    centroids = [fps[i] for i in idxs]
    return [
        assign_cluster_label(fp, centroids, idxs) for fp in fps
    ]

def group_mols(
    smis: Sequence[str], radius: int = 2, length: int = 2048,
    similarity: float = 0.35
) -> Tuple[Set[str], Set[str], Set[str]]:
    """group molecules based on their cluster membership

    The molecules are first clustered based on the input fingerprint and 
    similarity arguments. Then, they are grouped into three based on whether 
    they belong to the largest cluster, a mid-sized cluster, or a singleton cluster (a cluster containing no members beyond the centroid)
    
    Parameters
    ----------
    smis : Sequence[str]
        the SMILES strings corresponding to the molecules to group
    radius : int, default=2
        the radius of the fingerprint used to represent molecules
    length : int, default=2048
        the length of the fingerprint used to represent molecules
    similarity : float, default=0.35
        the maximum allowable similarity between cluster centers

    Returns
    -------
    large : Set[str]
        all molecules belonging to the largest cluster
    mids : Set[str]
        all molecules belonging to mid-sized clusters
    singletons : Set[str]
        all molecules belonging to singleton clusters
    """
    fps = [smi_to_fp(smi, radius, length) for smi in smis]
    labels = cluster_fps(fps, similarity)
    d_label_size = Counter(labels)

    largest_cluster_size = d_label_size.most_common()[0][1]

    large, mids, singletons = set(), set(), set()
    for smi, label in zip(smis, labels):
        cluster_size = d_label_size[label]
        if cluster_size == largest_cluster_size:
            large.add(smi)
        elif cluster_size == 1:
            singletons.add(smi)
        else:
            mids.add(smi)
    
    if len(large) > largest_cluster_size:
        print(f'large size: {len(large)}')
    elif len(large) < largest_cluster_size:
        print('you got bugs bro')

    if len(singletons) + len(mids) + len(large) != len(set(smis)):
        raise RuntimeError('not every molecule was assigned a cluster group')

    return large, mids, singletons

def recursive_conversion(nested_dict):
    if not isinstance(nested_dict, defaultdict):
        return nested_dict

    for k in nested_dict:
        sub_dict = nested_dict[k]
        nested_dict[k] = recursive_conversion(sub_dict)
    return dict(nested_dict)

def get_smis_from_data(p_data) -> Set:
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        smis = {row[0] for row in reader}
    
    return smis

def gather_experiment_cluster_results(
        experiment: Path, true_clusters,
    ) -> List[Tuple[float, float, float]]:
    experiment = Experiment(experiment)

    return [
        experiment.calculate_cluster_fraction(i, true_clusters)
        for i in range(experiment.num_iters)
    ]

def gather_prune_cluster_results(d_prune, true_clusters):
    Y = np.array([
        gather_experiment_cluster_results(rep, true_clusters)
        for rep in tqdm(d_prune.iterdir(), 'Reps', None, False)
    ])

    Y_mean = Y.mean(axis=0)
    Y_sd = np.sqrt(Y.var(axis=0))
    
    return {
        'large': np.column_stack((Y_mean[:, 0], Y_sd[:, 0])),
        'mids': np.column_stack((Y_mean[:, 1], Y_sd[:, 1])),
        'singletons': np.column_stack((Y_mean[:, 2], Y_sd[:, 2]))
    }

def gather_cluster_results(parent_dir, true_clusters, overwrite: bool = False):
    nested_dict = lambda: defaultdict(nested_dict)
    results = nested_dict()

    N = sum(len(cluster) for cluster in true_clusters)
    parent_dir = Path(parent_dir)
    cached_rewards = parent_dir / f'.all_rewards_{N}.pkl'
    if cached_rewards.exists() and not overwrite:
        return pickle.load(open(cached_rewards, 'rb'))

    for d_split in tqdm(parent_dir.iterdir(), 'Splits', None, False):
        if not d_split.is_dir():
            continue
        for d_batches in tqdm(d_split.iterdir(), 'Bs', None, False):
            for d_prune in tqdm(d_batches.iterdir(), 'Prune', None, False):
                results[
                    float(d_split.name)][
                    d_batches.name][
                    d_prune.name
                ] = gather_prune_cluster_results(d_prune, true_clusters)
    results = recursive_conversion(results)

    pickle.dump(results, open(cached_rewards, 'wb'))

    return results

################################################################################
#------------------------------------------------------------------------------#
################################################################################

METRICS = ['greedy', 'ucb', 'ts', 'ei', 'pi']
METRIC_NAMES = {'greedy': 'greedy', 'ucb': 'UCB', 'ts': 'TS',
                'ei': 'EI', 'pi': 'PI'}
METRIC_COLORS = dict(zip(METRICS, sns.color_palette('bright')))

MODELS = ['rf', 'nn', 'mpn']
MODEL_COLORS = dict(zip(MODELS, sns.color_palette('dark')))

SPLITS = [0.004, 0.002, 0.001]

DASHES = ['dash', 'dot', 'dashdot']
MARKERS = ['circle', 'square', 'diamond']

CLUSTERS = ('singletons', 'mids', 'large')

PRUNE = ('best', 'leader', 'maxmin', 'random')
PRUNE_COLORS = dict(zip(PRUNE, sns.color_palette('dark')))

def style_axis(ax):
    ax.set_xlabel(f'Molecules sampled')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.xaxis.set_tick_params(rotation=30)
    ax.grid(True)

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e3:
        return f'{x*1e-3:0.0f}k'

    return f'{x:0.0f}'

def plot_clusters(results, split: float, batches: str, size: int):
    fig, axs = plt.subplots(
        1, 3, sharex=True, sharey=True, figsize=(4/1.5 * 3, 4)
    )

    fmt = 'o-'
    ms = 5
    capsize = 2
    
    xs = [int(size*split * i) for i in range(1, 7)]

    for ax, cluster in zip(axs, CLUSTERS):
        for prune in results[split][batches]:
            Y = results[split][batches][prune][cluster]

            ax.errorbar(
                xs, Y[:,0], yerr=Y[:,1], color=PRUNE_COLORS[prune],
                fmt=fmt, ms=ms, mec='black', capsize=capsize,
                label=prune
            )

        formatter = ticker.FuncFormatter(abbreviate_k_or_M)
        ax.xaxis.set_major_formatter(formatter)
        
        ax.set_title(cluster.capitalize())
        ax.set_ylabel(f'Fraction found')
        style_axis(ax)
    axs[0].legend(loc='upper left', title='Strategy')
    
    fig.tight_layout()

    return fig

def write_figures(results, size, path):
    Path(path).mkdir(exist_ok=True, parents=True)

    for split in results:
        for batches in results[split]:
            plot_clusters(results, split, batches, size).savefig(
                f'{path}/s{split}_{batches}.png'
            )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--parent-dir',
                        help='the parent directory containing all of the results. NOTE: the directory must be organized in the folowing manner: <root>/<online,retrain>/<split_size>/<model>/<metric>/<repeat>/<run>. See the README for a visual description.')
    parser.add_argument('-t', '--true-pkl',
                        help='a pickle file containing a dictionary of the true scoring data as either a List[Tuple] or Dict')
    parser.add_argument('--minimize', action='store_true', default=False,
                        help='whether the objective for which you are calculating performance should be maximized.')
    parser.add_argument('--size', type=int,
                        help='the size of the full library which was explored. You only need to specify this if you are using a truncated pickle file. I.e., your pickle file contains only the top 1000 scores because you only intend to calculate results of the top-k, where k <= 1000')
    parser.add_argument('-N', type=int,
                        help='the number of top scores from which to calculate perforamnce')
    parser.add_argument('-r', '--radius', type=int, default=2)
    parser.add_argument('-l', '--length', type=int, default=2048)
    parser.add_argument('-s', '--similarity', type=float, default=0.35)
    parser.add_argument('--path', default='diversity-clusters')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='whether to overwrite the hidden cache file. This is useful if there is new data in PARENT_DIR.')

    args = parser.parse_args()

    if args.true_pkl:
        true_data = pickle.load(open(args.true_pkl, 'rb'))
        size = args.size or len(true_data)

        if isinstance(true_data, dict):
            true_data = sorted(true_data.items(), key=itemgetter(1))
        elif isinstance(true_data, Sequence):
            true_data = sorted(true_data, key=itemgetter(1))
        else:
            raise ValueError(
                'Bad pickle file provided! Object must be a list or a dict!'
            )

        if args.minimize:
            true_data = true_data[:args.N]
            true_data = [(smi, -score) for smi, score in true_data]
        else:
            true_data = true_data[-args.N:]

    true_smis, true_scores = zip(*true_data)
    true_clusters = group_mols(
        true_smis, args.radius, args.length, args.similarity
    )

    results = gather_cluster_results(
        args.parent_dir, true_clusters, args.overwrite
    )
    # pprint.pprint(results, compact=True)
    write_figures(results, size, args.path)