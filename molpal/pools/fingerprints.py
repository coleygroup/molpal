from concurrent.futures import ProcessPoolExecutor as Pool
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List, Set, Tuple, Type, TypeVar

import h5py
from tqdm import tqdm

from molpal.encoder import Encoder

T = TypeVar('T')

def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    """Consume an iterable in batches of size chunk_size"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

def feature_matrix_hdf5(xs: Iterable[T], size: int, *, ncpu: int = 0,
                        encoder: Type[Encoder] = Encoder(),
                        name: str = 'fps',
                        path: str = '.') -> Tuple[str, Set[int]]:
    """Precalculate the fature matrix of xs with the given encoder and store
    the matrix in an HDF5 file
    
    Parameters
    ----------
    xs: Iterable[T]
        the inputs for which to generate the feature matrix
    size : int
        the length of the iterable
    ncpu : int (Default = 0)
        the number of cores to parallelize feature matrix generation over
    encoder : Type[Encoder] (Default = Encoder('pair'))
        an object that encodes inputs from an identifier representation to
        a feature representation
    name : str (Default = 'fps')
        the name of the output HDF5 file
    path : str (Default = '.')
        the path under which the HDF5 file should be written

    Returns
    -------
    fps_h5 : str
        the filename of an hdf5 file containing the feature matrix of the
        representations generated from the molecules in the input file.
        The row ordering corresponds to the ordering of smis
    invalid_idxs : Set[int]
        the set of indices in xs containing invalid inputs
    """
    fps_h5 = str(Path(path)/f'{name}.h5')

    with Pool(max_workers=ncpu) as pool, h5py.File(fps_h5, 'w') as h5f:
        CHUNKSIZE = 512

        fps_dset = h5f.create_dataset(
            'fps', (size, len(encoder)),
            chunks=(CHUNKSIZE, len(encoder)), dtype='int8'
        )
        
        batch_size = CHUNKSIZE*ncpu
        n_batches = size//batch_size + 1

        invalid_idxs = set()
        i = 0
        offset = 0

        for xs_batch in tqdm(batches(xs, batch_size), total=n_batches,
                             desc='Precalculating fps', unit='batch'):
            fps = pool.map(encoder.encode_and_uncompress, xs_batch,
                           chunksize=2*ncpu)
            for fp in tqdm(fps, total=batch_size, smoothing=0., leave=False):
                while fp is None:
                    invalid_idxs.add(i+offset)
                    offset += 1
                    fp = next(fps)

                fps_dset[i] = fp
                i += 1
        # original dataset size included potentially invalid xs
        valid_size = size - len(invalid_idxs)
        if valid_size != size:
            fps_dset.resize(valid_size, axis=0)

    return fps_h5, invalid_idxs
