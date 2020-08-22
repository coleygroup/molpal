# from collections import defaultdict
import csv
import itertools as it
from itertools import chain
import multiprocessing as mp
import os
from pathlib import Path
from random import sample
import re
import sys
import timeit
from typing import Dict, List, Optional

import h5py
import numpy as np
from scipy import sparse
from sklearn import cluster

from .fingerprints import parse_smiles_par

try:
    MAX_CPU = len(os.sched_getaffinity(0))
except AttributeError:
    MAX_CPU = mp.cpu_count()

def cluster_fps_h5(fps_h5: str, ncluster: int = 100) -> List[int]:
    """Cluster the inputs represented by the feature matrix in fps_h5

    Parameters
    ----------
    fps : str
        the filepath of an HDF5 file containing the feature matrix of the
        molecules, where N is the number of molecules and M is the length of 
        the feature representation
    ncluster : int (Default = 100)
        the number of clusters to generate

    Returns
    -------
    cluster_ids : List[int]
        the cluster id corresponding to a given fingerprint
    """
    begin = timeit.default_timer()

    BATCH_SIZE = 1024
    n_iter = 1000

    clusterer = cluster.MiniBatchKMeans(n_clusters=ncluster,
                                        batch_size=BATCH_SIZE)

    with h5py.File(fps_h5, 'r') as h5f:
        fps = h5f['fps']

        for _ in range(n_iter):
            idxs = sorted(sample(range(len(fps)), BATCH_SIZE))
            clusterer.partial_fit(fps[idxs])

        cluster_ids = [clusterer.predict(fps[i:i+BATCH_SIZE])
                       for i in range(0, len(fps), BATCH_SIZE)]

    elapsed = timeit.default_timer() - begin
    print(f'Clustering took: {elapsed:0.3f}s')

    return list(chain(*cluster_ids))

def cluster_fps(fps: List[np.ndarray],
                ncluster: int = 100, method: str = 'minibatch',
                njobs: Optional[int] = None) -> np.ndarray:
    """Cluster the molecular fingerprints, fps, by a given method

    Parameters
    ----------
    fps : List[np.ndarray]
        a list of bit vectors corresponding to a given molecule's Morgan
        fingerprint (radius=2, length=1024)
    ncluster : int (Default = 100)
        the number of clusters to form with the given fingerprints (if the
        input method requires this parameter)
    method : str (Default = 'kmeans')
        the clusering method to use.
        Choices include:
        - k-means clustering: 'kmeans'
        - mini-batch k-means clustering: 'minibatch'
        - OPTICS clustering 'optics'
    njobs : Optional[int]
        the number of jobs to parallelize clustering over, if possible

    Returns
    -------
    cluster_ids : np.ndarray
        the cluster id corresponding to a given fingerprint
    """
    begin = timeit.default_timer()

    fps = sparse.vstack(fps, format='csr')

    if method == 'kmeans':
        clusterer = cluster.KMeans(n_clusters=ncluster, n_jobs=njobs)
    elif method == 'minibatch':
        clusterer = cluster.MiniBatchKMeans(n_clusters=ncluster, n_init=10,
                                            batch_size=100, init_size=1000)
    elif method == 'optics':
        clusterer = cluster.OPTICS(min_samples=0.01, metric='jaccard',
                                   n_jobs=njobs)
    else:
        raise ValueError(f'{method} is not a supported clustering method')

    cluster_ids = clusterer.fit_predict(fps)

    elapsed = timeit.default_timer() - begin
    print(f'Clustering and predictions took: {elapsed:0.3f}s')

    return cluster_ids

def write_clusters(path: str, d_cluster_smidxs: Dict[int, List[int]],
                   smis: h5py.Dataset) -> str:
    """Write each cluster to a separate file under the given path"""
    p = Path(path)
    if not p.is_dir():
        p.mkdir(parents=True)

    for _, smidxs in d_cluster_smidxs.items():
        write_cluster(str(p / f'cluster_{id:003d}.smi'), smidxs, smis)

def write_cluster(filepath: str, smidxs: List[int],
                  smis: h5py.Dataset) -> None:
    """Write each string in smis to filepath"""
    with open(filepath, 'w', newline='') as f:
        f.write('smiles\n')
        for idx in smidxs:
            smi = smis[idx]
            f.write(f'{smi}\n')

def cluster_smiles_file(filepath: str, delimiter: str = ',',
                        smiles_col: int = -1, title_line: bool = True,
                        njobs: int = -1, ncluster: int = 100,
                        path: str = '.') -> None:
    """
    Cluster the molecules in a .smi format file

    Parameters
    ----------
    filepath : str
        the filepath of the .smi file to cluster
    sep : str (Default = ' +|,|\t')
        an r-string used to separate fields in the file. By default, will
        match one or more spaces, a comma, or a tab as field separators
    smiles_col : int (Default = -1)
        the column containing the SMILES string in each row. If a value of -1
        is used, will use the first column containing a valid smiles string
    title_line : bool (Default = True)
        whether the file contains a title line
    njobs : int (Default = -1)
        the number of jobs to parallelize file parsing over. A value of -1
        uses all available cores, -2: all cores minus one, etc.
    ncluster : int (Default = 100)
        the number of clusters to group the molecules into
    out_path : Optional[str]
        the name of the directory to which all cluster files should be written.
        By default, will organize all clusters under the directory
        <path>/<filename>_clusters/
    """
    start = timeit.default_timer()

    fps_smis_h5 = parse_smiles_par(
        filepath, delimiter=delimiter, smiles_col=smiles_col,
        title_line=title_line, njobs=njobs)

    cluster_ids = cluster_fps_h5(fps_smis_h5, ncluster=ncluster)

    if path is None:
        p_smis = Path(filepath)
        p_clusters = p_smis.with_name(f'{p_smis.stem}_clusters')
        # path = Path(path) / f'{p_smis.stem}_clusters'

    with open(filepath) as fid_in, open(p_clusters, 'w') as fid_out:
        reader = csv.reader(fid_in, delimiter=delimiter)
        writer = csv.writer(fid_out)

        if title_line:
            next(reader)

        writer.write(['smiles', 'cluster'])

        for i, row in enumerate(reader):
            smi = row[smiles_col]
            writer.write([smi, cluster_ids[i]])

    elapsed = timeit.default_timer() - start
    print(f'Total time to cluster the {len(cluster_ids)} mols in "{filepath}"',
          f'over {njobs} CPUs: {elapsed:0.3f}s')

if __name__ == '__main__':
    cluster_smiles_file(sys.argv[1], njobs=MAX_CPU)
