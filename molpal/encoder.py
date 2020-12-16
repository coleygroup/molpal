"""This module contains the Encoder ABC and various implementations thereof.
Encoders transform input representations into (un)compressed representations
for use with clustering and model training/prediction."""

from abc import abstractmethod
from functools import partial
from typing import Optional, NoReturn, Text, Type, TypeVar
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

try:
    from map4 import map4
except ImportError:
    pass

T = TypeVar('T')            # input identifier
T_comp = TypeVar('T_comp')  # compressed feature representation of an input

class Encoder:
    """An Encoder implements methods to transforms an identifier into its
    compressed and uncompressed feature representations
    """
    def __init__(self, fingerprint: str = 'pair', radius: int = 2,
                 length: int = 2048, **kwargs):
        self.fingerprint = fingerprint
        self.length = length
        self.radius = radius

    def __call__(self, x: T) -> T_comp:
        return self.encode(x)

    def __len__(self) -> int:
        return self.length

    def encode(self, x: T) -> Optional[T_comp]:
        try:
            return self._encode(x, self.fingerprint, self.radius, self.length)
        except:
            return None
    
    @staticmethod
    def _encode(smi: str, fingerprint: str, radius: int, length: int) -> T_comp:
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
        if fingerprint == 'morgan':
            return rdmd.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=length, useChirality=True)

        if fingerprint == 'pair':
            return rdmd.GetHashedAtomPairFingerprintAsBitVect(
                mol, minLength=1, maxLength=1+radius, nBits=length)
        
        if fingerprint == 'rdkit':
            return rdmd.RDKFingerprint(
                mol, minPath=1, maxPath=1+radius, fpSize=length)
        
        if fingerprint == 'maccs':
            return rdmd.GetMACCSKeysFingerprint(mol)

        if fingerprint == 'map4':
            return map4.MAP4Calculator(
                dimensions=length, radius=radius, is_folded=True
            ).calculate(mol)

        raise NotImplementedError(f'Unrecognized fingerprint: "{fingerprint}"')

    @staticmethod
    def uncompress(x_comp: T_comp) -> np.ndarray:
        return np.array(x_comp)

    def encode_and_uncompress(self, x: T) -> Optional[np.ndarray]:
        """Generate the uncompressed representation of x"""
        try:
            return self.uncompress(self.encode(x))
        except:
            return None

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(' + 
                f'fingerprint={self.fingerprint}, ' +
                f'radius={self.radius}, length={self.length})')
