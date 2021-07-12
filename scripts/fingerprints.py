import argparse
import csv
from functools import partial
import gzip
from itertools import chain, islice
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Set, Tuple

import h5py
import numpy as np
import ray
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdmd
from tqdm import tqdm

try:
    if 'redis_password' in os.environ:
        ray.init(
            address=os.environ["ip_head"],#'auto',
            _node_ip_address=os.environ["ip_head"].split(":")[0], 
            _redis_password=os.environ['redis_password']
        )
    else:
        ray.init(address='auto')
except ConnectionError:
    ray.init(num_cpus=len(os.sched_getaffinity(0)))

def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    """Consume an iterable in batches of size chunk_size"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

def smi_to_fp(smi: str, fingerprint: str,
              radius: int = 2, length: int = 2048) -> Optional[np.ndarray]:
    """fingerprint functions must be wrapped in a static function
    so that they may be pickled for parallel processing
        
    Parameters
    ----------
    smi : str
        the SMILES string of the molecule to encode
    fingerprint : str
        the the type of fingerprint to generate
    radius : int
        the radius of the fingerprint
    length : int
        the length of the fingerprint
    
    Returns
    -------
    T_comp
        the compressed feature representation of the molecule
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    if fingerprint == 'morgan':
        fp = rdmd.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=length, useChirality=True)
    elif fingerprint == 'pair':
        fp = rdmd.GetHashedAtomPairFingerprintAsBitVect(
            mol, minLength=1, maxLength=1+radius, nBits=length)
    elif fingerprint == 'rdkit':
        fp = rdmd.RDKFingerprint(
            mol, minPath=1, maxPath=1+radius, fpSize=length)
    elif fingerprint == 'maccs':
        fp = rdmd.GetMACCSKeysFingerprint(mol)
    else:
        raise NotImplementedError(
            f'Unrecognized fingerprint: "{fingerprint}"')

    x = np.empty(len(fp))
    DataStructs.ConvertToNumpyArray(fp, x)
    return x

@ray.remote
def _smis_to_fps(smis: Iterable[str], fingerprint: str = 'pair',
                 radius: int = 2,
                 length: int = 2048) -> List[Optional[np.ndarray]]:
    fps = [smi_to_fp(smi, fingerprint, radius, length) for smi in smis]
    return fps

def smis_to_fps(smis: Iterable[str], fingerprint: str = 'pair',
                radius: int = 2,
                length: int = 2048) -> List[Optional[np.ndarray]]:
    """Calculate the Morgan fingerprint of each molecule in smis

    Parameters
    ----------
    smis : Iterable[str]
        the SMILES strings of the molecules
    radius : int, default=2
        the radius of the fingerprint
    length : int, default=2048
        the number of bits in the fingerprint

    Returns
    -------
    List
        a list of the corresponding morgan fingerprints in bit vector form
    """
    chunksize = int(ray.cluster_resources()['CPU'] * 64)
    refs = [
        _smis_to_fps.remote(smis_chunk, fingerprint, radius, length)
        for smis_chunk in batches(smis, chunksize)
    ]
    fps_chunks = [
        ray.get(r)
        for r in tqdm(refs, desc='Calculating fingerprints',
                      unit='chunk', leave=False)
    ]
    fps = list(chain(*fps_chunks))

    return fps

def fps_hdf5(smis: Iterable[str], size: int,
             fingerprint: str = 'pair', radius: int = 2, length: int = 2048,
             filepath: str = 'fps.h5') -> Tuple[str, Set[int]]:
    """Prepare an HDF5 file containing the feature matrix of the input SMILES
    strings

    Parameters
    ----------
    smis : Iterable[str]
        the SMILES strings from which to build the feature matrix
    size : int
        the total number of smiles strings
    fingerprint : str, default='pair'
        the type of fingerprint to calculate
    radius : int, default=2
        the "radius" of the fingerprint to calculate. For path-based
        fingerprints, this corresponds to the path length
    length : int, default=2048
        the length/number of bits in the fingerprint
    filepath : str, default='fps.h5'
        the filepath of the output HDF5 file

    Returns
    -------
    str
        the filepath of the output HDF5 file
    invalid_idxs : Set[int]
        the set of invalid indices in the input SMILES strings
    """
    with h5py.File(filepath, 'w') as h5f:
        CHUNKSIZE = 512

        fps_dset = h5f.create_dataset(
            'fps', (size, length),
            chunks=(CHUNKSIZE, length), dtype='int8'
        )
        
        batchsize = 262144
        n_batches = size//batchsize + 1

        invalid_idxs = set()
        i = 0
        offset = 0

        for smis_batch in tqdm(batches(smis, batchsize), total=n_batches,
                               desc='Precalculating fps', unit='batch'):
            fps = smis_to_fps(smis_batch, fingerprint, radius, length)
            for fp in tqdm(fps, total=batchsize, leave=False, desc='Writing'):
                if fp is None:
                    invalid_idxs.add(i + offset)
                    offset += 1
                    continue

                fps_dset[i] = fp
                i += 1

        valid_size = size - len(invalid_idxs)
        if valid_size != size:
            fps_dset.resize(valid_size, axis=0)

    return filepath, invalid_idxs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='.',
                        help='the path under which to write the fingerprints file')
    parser.add_argument('--name',
                        help='what to name the fingerprints file. If no suffix is provided, will add ".h5". If no name is provided, output file will be name <library>.h5')
    parser.add_argument('--fingerprint', default='pair',
                        choices={'morgan', 'rdkit', 'pair', 'maccs', 'map4'},
                        help='the type of encoder to use')
    parser.add_argument('--radius', type=int, default=2,
                        help='the radius or path length to use for fingerprints')
    parser.add_argument('--length', type=int, default=2048,
                        help='the length of the fingerprint')

    parser.add_argument('--library', required=True, metavar='LIBRARY_FILEPATH',
                        help='the file containing members of the MoleculePool')
    parser.add_argument('--no-title-line', action='store_true', default=False,
                        help='whether there is no title line in the library file')
    parser.add_argument('--total-size', type=int,
                        help='the total number of molecules in the library file')
    parser.add_argument('--delimiter', default=',',
                        help='the column separator in the library file')
    parser.add_argument('--smiles-col', default=0, type=int,
                        help='the column containing the SMILES string in the library file')
    args = parser.parse_args()
    args.title_line = not args.no_title_line

    print(args)
    
    if args.name is None:
        args.name = Path(args.library).stem
    
    filepath = Path(args.path) / args.name
    if Path(filepath).suffix == '.h5':
        pass
    else:
        filepath.with_suffix('.h5')
        
    if Path(args.library).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open

    print('Precalculating feature matrix ...', end=' ')
    with open_(args.library) as fid:
        reader = csv.reader(fid, delimiter=args.delimiter)
        if args.total_size is None:
            total_size = sum(1 for _ in fid)
            fid.seek(0)
        else:
            total_size = args.total_size
            
        if args.title_line:
            total_size -= 1; next(reader)

        smis = (row[args.smiles_col] for row in reader)
        fps, invalid_lines = fps_hdf5(
            smis, total_size, args.fingerprint,
            args.radius, args.length, filepath
        )

    print('Done!')
    print(f'Feature matrix was saved to "{fps}"', flush=True)

    if len(invalid_lines) == 0:
        print('Detected no invalid lines! When using this fingerprints file,',
              'you should add "--invalid-lines" to the command line or',
              '"invalid-lines=[] to the configuration file to speed up',
              'pool construction')

if __name__ == "__main__":
    main()
