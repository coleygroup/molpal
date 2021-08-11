"""This module contains the Feautrizer dataclass. A Featurizer contains fields
releveant to calcaulting the feature matrix of a list of SMILES strings"""
from dataclasses import dataclass
from itertools import chain, islice
import math
from typing import Iterable, Iterator, List, Optional, TypeVar

import numpy as np
import ray
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

try:
    from map4 import map4
except ImportError:
    pass

T = TypeVar('T')            # input identifier
T_comp = TypeVar('T_comp')  # compressed feature representation of an input

@dataclass
class Featurizer:
    fingerprint: str = 'pair'
    radius: int = 2
    length: int = 2048

    def __len__(self):
        return 167 if self.fingerprint == 'maccs' else self.length 

    def __call__(self, smi: str) -> Optional[np.ndarray]:
        return featurize(smi, self.fingerprint, self.radius, self.length)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            + f'fingerprint={self.fingerprint}, '
            + f'radius={self.radius}, ' 
            + f'length={self.length})'
        )

def featurize(smi, fingerprint, radius, length) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    if fingerprint == 'morgan':
        fp = rdmd.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=length, useChirality=True
        )
    elif fingerprint == 'pair':
        fp = rdmd.GetHashedAtomPairFingerprintAsBitVect(
            mol, minLength=1, maxLength=1+radius, nBits=length
        )
    elif fingerprint == 'rdkit':
        fp = rdmd.RDKFingerprint(
            mol, minPath=1, maxPath=1+radius, fpSize=length
        )
    elif fingerprint == 'maccs':
        fp = rdmd.GetMACCSKeysFingerprint(mol)
    elif fingerprint == 'map4':
        fp = map4.MAP4Calculator(
            dimensions=length, radius=radius, is_folded=True
        ).calculate(mol)
    else:
        raise NotImplementedError(f'Unrecognized fingerprint: "{fingerprint}"')

    X = np.empty(len(fp))
    ConvertToNumpyArray(fp, X)
    return X

def chunks(it: Iterable[T], chunk_size: int) -> Iterator[List]:
    """Batch an iterable into batches of size chunk_size, with the final
    batch potentially being smaller"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

def feature_matrix(smis, featurizer: Featurizer = Featurizer()):
    fingerprint = featurizer.fingerprint
    radius = featurizer.radius
    length = featurizer.length

    @ray.remote
    def featurize_chunk(smis, fingerprint, radius, length):
        return [featurize(smi, fingerprint, radius, length) for smi in smis]

    chunksize = int(math.sqrt(ray.cluster_resources()['CPU']) * 512)
    refs = [
        featurize_chunk.remote(smis_chunk, fingerprint, radius, length)
        for smis_chunk in chunks(smis, chunksize)
    ]
    fps_chunks = [
        ray.get(r)
        for r in tqdm(refs, desc='Featurizing', unit='smi', leave=False)
    ]
    fps = list(chain(*fps_chunks))
    return np.array(fps)