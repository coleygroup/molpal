""" Tries out a few different clustering algorithms in sklearn and reports speed """

from sklearn import cluster
from scipy import sparse
from molpal.pools import MoleculePool
from molpal.featurizer import Featurizer

import numpy as np 
import timeit

methods = [
    'kmeans', 
    'minibatch',]

featurizer = Featurizer(
    fingerprint='pair',
    radius=2,
    length=2048,
)

pool = MoleculePool(
    libraries=['data/drd2_data.csv'],
    smiles_col=0,
    fps='data/drd_fps.h5',
    featurizer=featurizer
)

fps = sparse.csr_matrix(np.array([fp for fp in pool.fps()]))
ncluster = 20000
ncpu = 8

for method in methods: 
    print(f'Method: {method}')
    begin = timeit.default_timer()

    if method =="kmeans":
        clusterer = cluster.KMeans(n_clusters=ncluster)
    elif method == "minibatch":
        clusterer = cluster.MiniBatchKMeans(
            n_clusters=ncluster, n_init=10, batch_size=100
        )
    cluster_ids = clusterer.fit_predict(fps)

    elapsed = timeit.default_timer() - begin
    print(f"Clustering and predictions took: {elapsed:0.3f}s")