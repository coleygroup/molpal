from argparse import ArgumentParser
from collections import Counter
from enum import auto, Enum
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Set, Sequence, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import ray
from ray.util.joblib import register_ray
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ExplicitBitVect
from rdkit.SimDivFilters import rdSimDivPickers
import seaborn as sns
from sklearn import cluster
import torch
from tqdm import tqdm

sys.path.append('../molpal')
from experiment import Experiment
from molpal.models.chemprop.data.data import (
    MoleculeDataset, MoleculeDatapoint, MoleculeDataLoader
)
from utils import (
    extract_smis, build_true_dict, chunk, style_axis, abbreviate_k_or_M
)

sns.set_theme(style='white', context='paper')

ray.init('auto')
register_ray()

class Representation(Enum):
    MORGAN = auto()
    EMBEDDING = auto()

def embeddings(
    smis: Iterable[str], experiment: Experiment, i: int
) -> np.ndarray:
    dataset = MoleculeDataset([MoleculeDatapoint([smi]) for smi in smis])
    data_loader = MoleculeDataLoader(
        dataset=dataset, batch_size=50, num_workers=8,
    )
    model = experiment.model(i)

    embeddings_batches = []
    with torch.no_grad():
        for componentss, _ in tqdm(data_loader):
            embeddings_batches.append(model.model.featurize(componentss))
        embeddings = torch.cat(embeddings_batches)

    return embeddings.cpu().numpy()

def cluster_embeddings(Z: np.ndarray) -> np.ndarray:
    for eps in np.linspace(0.1, 0.5, num=5):
        print(eps)
        labels = cluster.DBSCAN(
            eps, min_samples=1, metric='euclidean', n_jobs=-1
        ).fit_predict(Z)

        sizes = Counter(labels)
        counts = Counter(sizes.values())

        print(counts)

    return labels

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
    centroids: Sequence[ExplicitBitVect],
    labels: Sequence[int]
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

def cluster_fps(
    fps: Sequence[ExplicitBitVect], similarity: float = 0.35
) -> List[int]:
    """Cluster the input molecular fingerprints and return their
    associated labels

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

    return [assign_cluster_label(fp, centroids, idxs) for fp in fps]

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

def cluster_mols(
    smis: Iterable[str], representation: Representation, radius: int = 2, 
    length: int = 2048, similarity: float = 0.35,
    experiment: Optional[Union[str, Path]] = None
):
    if representation == Representation.MORGAN:
        fps = [smi_to_fp(smi, radius, length) for smi in smis]
        labels = cluster_fps(fps, similarity)
    elif representation == Representation.EMBEDDING:
        experiment = Experiment(args.experiment, d_smi_idx)
        Z = embeddings(top_N_smis, experiment, 2)
        labels = cluster_embeddings(Z)

    return labels

if __name__ == "__main__":
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
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('-r', '--representation',
                        help='the number of top scores from which to calculate the reward')
    args = parser.parse_args()
    args.title_line = not args.no_title_line

    representation = Representation[args.representation.upper()]

    smis = extract_smis(args.library, args.smiles_col, args.title_line)
    d_smi_idx = {smi: i for i, smi in enumerate(smis)}

    d_smi_score = build_true_dict(
        args.true_csv, args.smiles_col, args.score_col,
        args.title_line, args.maximize
    )

    true_smis_scores = sorted(d_smi_score.items(), key=lambda kv: kv[1])[::-1]
    true_top_N = true_smis_scores[:args.N]
    top_N_smis, top_N_scores  = zip(*true_top_N)

    radius, length, similarity = 2, 2048, 0.35
    labels = cluster_mols(smis, representation, radius, length, similarity)
    # experiment = Experiment(args.experiment, d_smi_idx)
    # Z = embeddings(top_N_smis, experiment, 2)
