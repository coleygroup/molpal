from argparse import ArgumentParser
from collections import Counter
from enum import auto, Enum
import sys
from typing import (
    Iterable, List, Mapping, Set, Sequence, Tuple, Union
)

import joblib
from matplotlib import pyplot as plt
import numpy as np
import ray
from ray.util.joblib import register_ray
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ExplicitBitVect
from rdkit.SimDivFilters import rdSimDivPickers
import seaborn as sns
import sklearn.cluster
import torch
from tqdm import tqdm

sys.path.append('../molpal')
from experiment import Experiment
from molpal.models.chemprop.data.data import (
    MoleculeDataset, MoleculeDatapoint, MoleculeDataLoader
)
from utils import (extract_smis, build_true_dict)

sns.set_theme(style='white', context='paper')

ray.init('auto')
register_ray()

Fingerprint = Union[ExplicitBitVect]

class Representation(Enum):
    MORGAN = auto()
    EMBEDDING = auto()

class ClusterMode(Enum):
    DBSCAN = auto()
    DISE = auto()

def embeddings(smis: Iterable[str], model) -> np.ndarray:
    dataset = MoleculeDataset([MoleculeDatapoint([smi]) for smi in smis])
    data_loader = MoleculeDataLoader(
        dataset=dataset, batch_size=50, num_workers=8,
    )

    Z_batches = []
    with torch.no_grad():
        for componentss, _ in tqdm(data_loader, leave=False):
            Z_batches.append(model.featurize(componentss))
        Z = torch.cat(Z_batches)

    return Z.cpu().numpy()

def smi_to_fp(smi: str, radius: int, length: int):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smi), radius, length, useChirality=True
    )

def compute_DISE_centroids(fps, similarity: float = 0.35) -> List[int]:
    distance = 1. - similarity
    lp = rdSimDivPickers.LeaderPicker()
    idxs = list(lp.LazyBitVectorPick(fps, len(fps), distance))

    return idxs

def assign_cluster_label(fp, centroids, labels) -> int:
    i = np.array(
        DataStructs.BulkTanimotoSimilarity(fp, centroids)
    ).argmax()

    return labels[i]

def cluster_fps_DISE(fps, similarity: float = 0.35) -> List[int]:
    """Cluster the input molecular fingerprints and return their
    associated labels using the DIrected Sphere Exclusion algorithm

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
    idxs = compute_DISE_centroids(fps, similarity)
    centroids = [fps[i] for i in idxs]

    labels = [assign_cluster_label(fp, centroids, idxs) for fp in fps]

    return labels

def cluster_array(
    X: np.ndarray, cluster_mode: ClusterMode, eps=0.2, metric='euclidean'
) -> np.ndarray:

    with joblib.parallel_backend('ray'):
        clustering = {
            ClusterMode.DBSCAN: sklearn.cluster.DBSCAN(
                eps, min_samples=1, metric=metric, n_jobs=-1
            ),
        }[cluster_mode]

        labels = clustering.fit_predict(X)

    return labels

def featurize_mols(
    smis: Iterable[str], representation: Representation,
    radius: int = 2, length: int = 2048, model = None,
) -> Union[np.ndarray, List[Fingerprint]]:
    if representation == Representation.MORGAN:
        X = [smi_to_fp(smi, radius, length) for smi in tqdm(smis)]
    elif representation == Representation.EMBEDDING:
        X = embeddings(smis, model)

    return X

def cluster_mols(
    X_or_fps: Union[np.ndarray, List[Fingerprint]],
    representation: Representation,
    cluster_mode: ClusterMode, similarity: float = 0.35
):
    if representation != Representation.EMBEDDING:
        fps = X_or_fps
        X = np.empty((len(fps), len(fps[0])), dtype=bool)
        [DataStructs.ConvertToNumpyArray(fp, x) for fp, x in zip(fps, X)]
        metric = 'jaccard'
        eps = 0.65
    else:
        X = X_or_fps
        metric = 'euclidean'
        eps = 0.4

    if cluster_mode == ClusterMode.DISE:
        labels = cluster_fps_DISE(fps, similarity)
    else:
        labels = cluster_array(X, cluster_mode, eps, metric)

    print(eps, Counter(Counter(labels).values()))
    return labels

def group_clusters(
    smis: Sequence[str], labels: Sequence
) -> Tuple[Set[str], Set[str], Set[str]]:
    """group molecules based on their cluster membership
    
    Molecules are lumped into into three groups based on whether they belong to 
    the largest cluster, a mid-sized cluster, or a singleton cluster (a cluster 
    containing no members beyond the centroid)
    
    Parameters
    ----------
    smis : Sequence[str]
        the SMILES strings corresponding to the molecules to group
    labels : Sequence
        a parallel sequence of cluster labels for each molecule

    Returns
    -------
    large : Set[str]
        all molecules belonging to the largest cluster
    mids : Set[str]
        all molecules belonging to mid-sized clusters
    singletons : Set[str]
        all molecules belonging to singleton clusters
    """
    d_label_size = Counter(labels)
    largest_cluster_size = d_label_size.most_common(1)[1]

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
        raise RuntimeError(
            f'too many molecules assigned to largest cluster: {len(large)}'
        )
    if len(large) < largest_cluster_size:
        raise RuntimeError(
            f'too few molecules assigned to largest cluster: {len(large)}'
        )
    if len(singletons) + len(mids) + len(large) != len(set(smis)):
        raise RuntimeError('not every molecule was assigned a cluster group')

    return large, mids, singletons

def array_from_fps(fps: Sequence[Fingerprint]) -> np.ndarray:
    X = np.empty((len(fps), len(fps[0])), dtype=bool)
    [DataStructs.ConvertToNumpyArray(fp, x) for fp, x in zip(fps, X)]
    return X

def plot_cluster_sizes(d_label_sizes: Sequence[Mapping], eps: float):
    N = len(d_label_sizes)
    fig, axs = plt.subplots(
        N, 1, figsize=(5, 1.2*N), sharex=True, sharey=True
    )

    axs = [axs] if N == 1 else axs

    for i, d_label_size in enumerate(d_label_sizes):
        axs[i].hist(
            d_label_size.values(), bins=np.arange(0, 1000, 10),
            log=True, edgecolor='none'
        )
        if N > 1:
            axs[i].set_ylabel(i+1, rotation=0)
        axs[i].grid(True)
            
    axs[-1].set_xlabel(f'Cluster size (eps={eps:0.2f})')

    fig.tight_layout()
    return fig

def main():
    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment', '--expt',
                        help='the top-level directory generated by the MolPAL run. I.e., the directory with the "data" and "chkpts" directories')
    parser.add_argument('-l', '--library',
                        help='the library file used for the corresponding MolPAL run.')
    parser.add_argument('--true-csv',
                        help='a pickle file containing a dictionary of the true scoring data')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('--no-title-line', action='store_true', default=False)
    parser.add_argument('--maximize', action='store_true', default=False,
                        help='whether the objective for which you are calculating performance should be maximized.')
    parser.add_argument('-N', type=int,
                        help='the number of top scores from which to calculate the reward')
    parser.add_argument('--name',
                        help='the filepath to which the plot should be saved')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('-r', '--representation')
    parser.add_argument('-cm', '--cluster-mode')
    parser.add_argument('--eps', type=float, default=0.35)
    parser.add_argument('--metric', default='euclidean')
    parser.add_argument('--similarity', type=float, default=0.35)

    args = parser.parse_args()
    args.title_line = not args.no_title_line

    representation = Representation[args.representation.upper()]
    cluster_mode = ClusterMode[args.cluster_mode.upper()]

    smis = extract_smis(args.library, args.smiles_col, args.title_line)
    d_smi_idx = {smi: i for i, smi in enumerate(smis)}

    d_smi_score = build_true_dict(
        args.true_csv, args.smiles_col, args.score_col,
        args.title_line, args.maximize
    )

    true_smis_scores = sorted(d_smi_score.items(), key=lambda kv: kv[1])[::-1]
    true_top_N = true_smis_scores[:args.N]
    top_N_smis, _  = zip(*true_top_N)

    radius, length, similarity = 2, 2048, 0.35
    expt = Experiment(args.experiment, d_smi_idx)

    if representation == Representation.EMBEDDING:
        ds = []
        for i in range(1, 6):
            model = expt.model(i).model
            X = embeddings(top_N_smis, model)
            labels = cluster_array(X, cluster_mode, args.eps, args.metric)

            d_label_size = Counter(labels)
            ds.append(d_label_size)
            print(args.eps, Counter(d_label_size.values()))

        plot_cluster_sizes(ds, args.eps).savefig(args.name, dpi=args.dpi)
    else:
        fps = [smi_to_fp(smi, radius, length) for smi in tqdm(top_N_smis)]
        if cluster_mode != ClusterMode.DISE:
            X = array_from_fps(fps)
            labels = cluster_array(X, cluster_mode, 1.-args.eps, 'jaccard')
        else:
            labels = cluster_fps_DISE(fps, similarity)

        d_label_size = Counter(labels)
        plot_cluster_sizes(
            [d_label_size], 1.-args.eps
        ).savefig(args.name, dpi=args.dpi)

        print(args.eps, Counter(d_label_size.values()))

if __name__ == "__main__":
    main()