"""This module contains functions for clutering a set of molecules"""
from collections import defaultdict
import csv
from itertools import chain
import os
from pathlib import Path
from random import sample
import sys
import timeit
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans

from molpal.objectives.pyscreener.postprocessing import fingerprints

def cluster(d_smi_score: Dict[str, Optional[float]],
            name: str = 'clusters', path: str = '.', **kwargs) -> List[Dict]:
    d_smi_cid = cluster_smis(d_smi_score.keys(), len(d_smi_score), **kwargs)

    clusters_csv = (Path(path)/f'{name}_clusters').with_suffix('.csv')
    with open(clusters_csv, 'w') as fid:
        writer = csv.writer(fid)
        writer.writerow(['smiles', 'cluster_id'])
        writer.writerows(d_smi_cid.items())

    d_cluster_smi_score = defaultdict(dict)
    for smi, score in d_smi_score.items():
        cid = d_smi_cid[smi]
        d_cluster_smi_score[cid][smi] = score

    return list(d_cluster_smi_score.values())

def cluster_smis(smis: Iterable[str], n_mols: int, *,
                 n_cluster: int = 10, 
                 path: str = '.', name: str = 'fps', 
                 **kwargs) -> Dict[str, int]:
    """Cluster the SMILES strings

    Parameters
    ----------
    smis : Iterable[str]
        the SMILES strings to cluster
    n_cluster : int (Default = 100)
        the number of clusters to generate
    path : str (Default = '.')
        the path under which to write the fingerprint file
    name : str (Default = '.')
        the name of the output fingerprint file
    **kwargs
        keyword arguments to fingerprints.gen_fps_h5

    Returns
    -------
    d_smi_cid : Dict[str, int]
        a mapping from SMILES string to cluster ID
    
    See also
    --------
    fingerprints.gen_fps_h5
    """
    fps_h5, invalid_idxs = fingerprints.gen_fps_h5(
        smis, n_mols, path=path, name=name, **kwargs)

    smis = [smi for i, smi in enumerate(smis) if i not in invalid_idxs]
    cids = cluster_fps_h5(fps_h5, n_cluster)
    
    return dict(zip(smis, cids))

def cluster_fps_h5(fps_h5: str, n_cluster: int = 10) -> List[int]:
    """Cluster the feature matrix of fingerprints in fps_h5

    Parameters
    ----------
    fps : str
        the filepath of an h5py file containing the NxM matrix of
        molecular fingerprints, where N is the number of molecules and
        M is the length of the fingerprint (feature representation)
    ncluster : int (Default = 100)
        the number of clusters to form with the given fingerprints (if the
        input method requires this parameter)

    Returns
    -------
    cids : List[int]
        the cluster id corresponding to a given fingerprint
    """
    begin = timeit.default_timer()

    with h5py.File(fps_h5, 'r') as h5f:
        fps = h5f['fps']
        chunk_size = fps.chunks[0]

        ITER = 1000
        BATCH_SIZE = min(1000, len(fps))
        clusterer = MiniBatchKMeans(n_clusters=n_cluster, batch_size=BATCH_SIZE)
        
        for _ in range(ITER):
            rand_idxs = sorted(sample(range(len(fps)), BATCH_SIZE))
            batch_fps = fps[rand_idxs]
            clusterer.partial_fit(batch_fps)

        cidss = [clusterer.predict(fps[i:i+chunk_size])
                 for i in range(0, len(fps), chunk_size)]

    elapsed = timeit.default_timer() - begin

    print(f'Clustering took: {elapsed:0.3f}s')

    return list(chain(*cidss))
