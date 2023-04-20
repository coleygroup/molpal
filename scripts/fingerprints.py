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
    if "redis_password" in os.environ:
        ray.init(
            address=os.environ["ip_head"],
            _node_ip_address=os.environ["ip_head"].split(":")[0],
            _redis_password=os.environ["redis_password"],
        )
    else:
        ray.init(address="auto")
except ConnectionError:
    ray.init(num_cpus=len(os.sched_getaffinity(0)))


def get_smis(
    libaries: Iterable[str], title_line: bool = True, delimiter: str = ",", smiles_col: int = 0
) -> Iterator[str]:
    for library in libaries:
        if Path(library).suffix == ".gz":
            open_ = partial(gzip.open, mode="rt")
        else:
            open_ = open

        with open_(library) as fid:
            reader = csv.reader(fid, delimiter=delimiter)
            if title_line:
                next(reader)

            for row in reader:
                yield row[smiles_col]


def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    """Consume an iterable in batches of size chunk_size"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])


@ray.remote
def _smis_to_mols(smis: Iterable) -> List[Optional[Chem.Mol]]:
    return [Chem.MolFromSmiles(smi) for smi in smis]


def smis_to_mols(smis: Iterable[str]) -> List[Optional[Chem.Mol]]:
    chunksize = int(ray.cluster_resources()["CPU"]) * 2
    refs = [_smis_to_mols.remote(smis_chunk) for smis_chunk in batches(smis, chunksize)]
    mols_chunks = [ray.get(r) for r in refs]
    return list(chain(*mols_chunks))


@ray.remote
def _mols_to_fps(
    mols: Iterable[Chem.Mol], fingerprint: str = "pair", radius: int = 2, length: int = 2048
) -> np.ndarray:
    """fingerprint functions must be wrapped in a static function
    so that they may be pickled for parallel processing

    Parameters
    ----------
    mols : Iterable[Chem.Mol]
        the molecules to encode
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
    if fingerprint == "morgan":
        fps = [
            rdmd.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, useChirality=True)
            for mol in mols
        ]
    elif fingerprint == "pair":
        fps = [
            rdmd.GetHashedAtomPairFingerprintAsBitVect(
                mol, minLength=1, maxLength=1 + radius, nBits=length
            )
            for mol in mols
        ]
    elif fingerprint == "rdkit":
        fps = [
            rdmd.RDKFingerprint(mol, minPath=1, maxPath=1 + radius, fpSize=length) for mol in mols
        ]
    elif fingerprint == "maccs":
        fps = [rdmd.GetMACCSKeysFingerprint(mol) for mol in mols]
    else:
        raise NotImplementedError(f'Unrecognized fingerprint: "{fingerprint}"')

    X = np.empty((len(mols), length))
    [DataStructs.ConvertToNumpyArray(fp, x) for fp, x in zip(fps, X)]

    return X


def mols_to_fps(
    mols: Iterable[Chem.Mol], fingerprint: str = "pair", radius: int = 2, length: int = 2048
) -> np.ndarray:
    """Calculate the Morgan fingerprint of each molecule

    Parameters
    ----------
    mols : Iterable[Chem.Mol]
        the molecules
    radius : int, default=2
        the radius of the fingerprint
    length : int, default=2048
        the number of bits in the fingerprint

    Returns
    -------
    List
        a list of the corresponding morgan fingerprints in bit vector form
    """
    chunksize = int(ray.cluster_resources()["CPU"] * 16)
    refs = [
        _mols_to_fps.remote(mols_chunk, fingerprint, radius, length)
        for mols_chunk in batches(mols, chunksize)
    ]
    fps_chunks = [
        ray.get(r) for r in tqdm(refs, desc="Calculating fingerprints", unit="chunk", leave=False)
    ]

    return np.vstack(fps_chunks)


def fps_hdf5(
    smis: Iterable[str],
    size: int,
    fingerprint: str = "pair",
    radius: int = 2,
    length: int = 2048,
    filepath: str = "fps.h5",
) -> Tuple[str, Set[int]]:
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
    with h5py.File(filepath, "w") as h5f:
        CHUNKSIZE = 1024

        fps_dset = h5f.create_dataset(
            "fps", (size, length), chunks=(CHUNKSIZE, length), dtype="int8"
        )

        batch_size = 4 * CHUNKSIZE * int(ray.cluster_resources()["CPU"])
        n_batches = size // batch_size + 1

        invalid_idxs = set()
        i = 0

        for smis_batch in tqdm(
            batches(smis, batch_size),
            total=n_batches,
            desc="Precalculating fps",
            unit="batch",
            unit_scale=batch_size,
        ):
            mols = smis_to_mols(smis_batch)
            invalid_idxs.update({i + j for j, mol in enumerate(mols) if mol is None})
            fps = mols_to_fps([mol for mol in mols if mol is not None], fingerprint, radius, length)

            fps_dset[i : i + len(fps)] = fps
            i += len(mols)

        valid_size = size - len(invalid_idxs)
        if valid_size != size:
            fps_dset.resize(valid_size, axis=0)

    return filepath, invalid_idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=Path, help="the filepath under of the output fingerprints HDF5 file. If no suffix is provided, will add '.h5'. If no name is provided, output file will be named <library>.h5"
    )
    parser.add_argument(
        "--fingerprint",
        default="pair",
        choices={"morgan", "rdkit", "pair", "maccs"},
        help="the type of encoder to use",
    )
    parser.add_argument(
        "--radius", type=int, default=2, help="the radius or path length to use for fingerprints"
    )
    parser.add_argument("--length", type=int, default=2048, help="the length of the fingerprint")
    parser.add_argument(
        "-l",
        "--library",
        required=True,
        nargs="+",
        help="the files containing members of the MoleculePool",
    )
    parser.add_argument(
        "--no-title-line",
        action="store_true",
        help="whether there is no title line in the library file",
    )
    parser.add_argument(
        "--total-size",
        type=int,
        help="(if known) the total number of molecules in the library file"
    )
    parser.add_argument(
        "-d", "--delimiter", default=",", help="the column separator in the library file"
    )
    parser.add_argument(
        "--smiles-col",
        default=0,
        type=int,
        help="the column containing the SMILES string in the library file",
    )
    args = parser.parse_args()
    args.title_line = not args.no_title_line
    path = (args.output or Path(args.library[0])).with_suffix(".h5")
    if args.total_size is None:
        args.total_size = sum(
            1 for _ in get_smis(args.library, args.title_line, args.delimiter, args.smiles_col)
        )

    print("Precalculating feature matrix ...", end=" ")

    smis = get_smis(args.library, args.title_line, args.delimiter, args.smiles_col)
    fps, invalid_lines = fps_hdf5(
        smis, args.total_size, args.fingerprint, args.radius, args.length, path
    )

    print("Done!")
    print(f'Feature matrix was saved to "{fps}"', flush=True)
    print(
        "When using this fingerprints file, you should add "
        f'"--fps {path} --invalid-lines {" ".join(invalid_lines)}" to the command line '
        "or the config file to speed up pool construction"
    )


if __name__ == "__main__":
    main()
