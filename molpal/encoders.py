"""This module contains the Encoder ABC and various implementations thereof.
Encoders transform input representations into (un)compressed representations
for use with clustering and model training/prediction."""

from abc import abstractmethod
from functools import partial
from typing import NoReturn, Text, Type, TypeVar
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np
import rdkit.Chem.AllChem as Chem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

try:
    from map4 import map4
except ImportError:
    pass

T = TypeVar('T')            # the input identifier
comp_T = TypeVar('comp_T')  # the compressed feature representation of an input
smi = str

def encoder(encoder: str, **kwargs):
    """Encoder factory function"""
    return {
        'morgan': MorganFingerprinter,
        'rdkit': RDKFingerprinter,
        'pair': AtomPairFingerprinter,
        'maccs': MACCSFingerprinter,
        'map4': MAP4Fingerprinter
    }.get(encoder, MorganFingerprinter)(**kwargs)


class Encoder(Protocol[T, comp_T]):
    """An Encoder implements methods to transforms an identifier into its
    compressed and uncompressed feature representations

    T: the type of the identifier
    comp_T: the type of the compressed feature representation

    This is an abstract class and cannot be instantiated by itself.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x: T) -> comp_T:
        return self.encode(x)

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the uncompressed feature representation"""

    @abstractmethod
    def encode(self, x: T) -> comp_T:
        """Generate the compressed representation of x"""

    @abstractmethod
    def write(self, x: T) -> Text:
        """Write the compressed representation of smi as text"""

    @classmethod
    @abstractmethod
    def read(self, rep: Text) -> comp_T:
        """Read in the textual form of a compressed representation"""

    @classmethod
    @abstractmethod
    def uncompress(self, x_comp: comp_T) -> np.ndarray:
        """Uncompress the compressed representation of x"""

    def encode_and_uncompress(self, x: T) -> np.ndarray:
        """Generate the uncompressed representation of x"""
        return self.uncompress(self.encode(x))

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'


class MorganFingerprinter(Encoder[smi, ExplicitBitVect]):
    """Encode SMILES strings into ExplicitBitVects using the Morgan
    fingerprint"""
    def __init__(self, radius: int = 2, length: int = 2048, **kwargs):
        self._encode = partial(Chem.GetMorganFingerprintAsBitVect,
                               radius=radius, nBits=length, useChirality=True)
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return self._encode.keywords.get('nBits')

    def encode(self, x: smi) -> ExplicitBitVect:
        return self._encode(Chem.MolFromSmiles(x))

    def write(self, x: smi) -> bytes:
        return self.encode(x).ToBinary()

    @classmethod
    def read(cls, rep: bytes) -> ExplicitBitVect:
        return ExplicitBitVect(rep)

    @classmethod
    def uncompress(cls, x_comp: ExplicitBitVect) -> np.ndarray:
        return np.array(x_comp)


class RDKFingerprinter(Encoder[smi, ExplicitBitVect]):
    """Encode SMILES strings into ExplicitBitVects using the RDKit
    fingerprint"""
    def __init__(self, radius: int = 2, length: int = 2048, **kwargs):
        min_path = 1
        max_path = min_path + radius
        self._encode = partial(Chem.RDKFingerprint, minPath=min_path, 
                               maxPath=max_path, fpSize=length)
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return self._encode.keywords.get('fpSize')

    def encode(self, x: smi) -> ExplicitBitVect:
        return self._encode(Chem.MolFromSmiles(x))

    def write(self, x: smi) -> bytes:
        return self.encode(x).ToBinary()

    @classmethod
    def read(cls, rep: bytes) -> ExplicitBitVect:
        return ExplicitBitVect(rep)

    @classmethod
    def uncompress(cls, x_comp: ExplicitBitVect) -> np.ndarray:
        return np.array(x_comp)


class AtomPairFingerprinter(Encoder[smi, ExplicitBitVect]):
    """Encode SMILES strings into ExplicitBitVects using the Atom Pair
    fingerprint"""
    def __init__(self, radius: int = 2, length: int = 2048, **kwargs):
        min_length = 1
        max_length = min_length + radius
        self._encode = partial(Chem.GetHashedAtomPairFingerprintAsBitVect,
                               minLength=min_length, maxLength=max_length, 
                               nBits=length)
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return self._encode.keywords.get('nBits')

    def encode(self, x: smi) -> ExplicitBitVect:
        return self._encode(Chem.MolFromSmiles(x))

    def write(self, x: smi) -> bytes:
        return self.encode(x).ToBinary()

    @classmethod
    def read(cls, rep: bytes) -> ExplicitBitVect:
        return ExplicitBitVect(rep)

    @classmethod
    def uncompress(cls, x_comp: ExplicitBitVect) -> np.ndarray:
        return np.array(x_comp)


class MACCSFingerprinter(Encoder[smi, ExplicitBitVect]):
    """Encode SMILES strings into ExplicitBitVects using the MACCS key
    fingerprint"""
    def __init__(self, **kwargs):
        self._encode = Chem.GetMACCSKeysFingerprint
        super().__init__(**kwargs)

    def __len__(self) -> int:
        # MACCS fingerprints are a constant 167 bits long
        return 167

    def encode(self, x: smi) -> ExplicitBitVect:
        return self._encode(Chem.MolFromSmiles(x))

    def write(self, x: smi) -> bytes:
        return self.encode(x).ToBinary()

    @classmethod
    def read(cls, rep: bytes) -> ExplicitBitVect:
        return ExplicitBitVect(rep)

    @classmethod
    def uncompress(cls, x_comp: ExplicitBitVect) -> np.ndarray:
        return np.array(x_comp)


class MAP4Fingerprinter(Encoder[smi, np.ndarray]):
    """Encode SMILES strings into ndarrays using the MAP4 fingerprint as
    defined in: Capecchi et al., J. Cheminform., 2020, 12 (43)."""
    def __init__(self, radius: int = 2, length: int = 2048, **kwargs):
        self._encoder = map4.MAP4Calculator(dimensions=length, radius=radius,
                                            is_folded=True)
        self.length = length
        super().__init(**kwargs)
    
    def __len__(self) -> int:
        return self.length
    
    def encode(self, x: smi) -> np.ndarray:
        return self._encoder.calculate(Chem.MolFromSmiles(x))
    
    @classmethod
    def uncompress(cls, x_comp: ExplicitBitVect) -> np.ndarray:
        return x_comp
        
