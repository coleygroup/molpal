"""This module contains functions for filtering molecules from inputs based on 
a desired set of properties"""

import csv
from itertools import zip_longest
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm

def filter_ligands(ligands: str,
                   **kwargs) -> Tuple[List[str], Optional[List[str]]]:
    if isinstance(ligands, str):
        p_ligand = Path(ligands)

        if p_ligand.suffix == '.csv':
            return filter_csv(ligands, **kwargs)
        if p_ligand.suffix in {'.sdf', '.smi'}:
            return filter_supply(ligands, **kwargs)
        
        return [ligands], None

    elif isinstance(ligands, Sequence):
        return filter_smis(ligands, **kwargs)
    
    raise TypeError('argument "ligand" must be of type str or Sequence[str]!')

def filter_mols(mols: List[Chem.Mol], names: Optional[List[str]] = None,
                max_atoms: int = 1000, max_weight: float = 10000.,
                max_logP: float = 10.,
                **kwargs) -> Tuple[List[str], Optional[List[str]]]:
    """Filter a list of molecules according to input critera

    Parameters
    ----------
    mols : List[Chem.Mol]
        the molecules to filter
    names : Optional[List[str]] (Default = None)
        a parallel list of names corresponding to each molecule
    max_atoms : int (Default = 1000)
    max_weight : float (Default = 10000)
    max_logP : float (Default = 10)

    Returns
    -------
    smis_filtered : List[str]
        the SMILES strings corresponding to the filtered molecules
    names_filtered : Optional[List[str]]
        the names corresponding to the filtered molecules. None, if no names
        were originally supplied
    """
    names = names or []
    
    smis_filtered = []
    names_filtered = []

    if names:
        for mol, name in tqdm(zip(mols, names), total=len(mols),
                            desc='Filtering mols', unit='mol'):
            if mol.GetNumHeavyAtoms() > max_atoms:
                continue

            props = QED.properties(mol)
            if props.MW > max_weight:
                continue
            if props.ALOGP > max_logP:
                continue

            smis_filtered.append(mol_to_smi(mol))
            names_filtered.append(name)
    else:
        names_filtered = None
        for mol in tqdm(mols, total=len(mols),
                        desc='Filtering mols', unit='mol'):
            if mol.GetNumHeavyAtoms() > max_atoms:
                continue

            props = QED.properties(mol)
            if props.MW > max_weight:
                continue
            if props.ALOGP > max_logP:
                continue

            smis_filtered.append(mol_to_smi(mol))

    return smis_filtered, names_filtered

def filter_smis(smis: List[str], names: Optional[List[str]] = None,
                **kwargs) -> Tuple[List[str], Optional[List[str]]]:
    mols = [mol_from_smi(smi)
            for smi in tqdm(smis, desc='Reading in mols', unit='mol')]

    return filter_mols(mols, names, **kwargs)

def filter_csv(csvfile: str, title_line: bool = True,
               smiles_col: int = 0, name_col: Optional[int] = None,
               **kwargs) -> Tuple[List[str], Optional[List[str]]]:
    with open(csvfile) as fid:
        reader = csv.reader(fid)
        if title_line:
            next(reader)

        reader = tqdm(reader, desc='Reading in mols', unit='mol')
        if name_col is None:
            mols = [mol_from_smi(row[smiles_col]) for row in reader]
            names = None
        else:
            mols_names = [(mol_from_smi(row[smiles_col]), row[name_col]) 
                           for row in reader]
            mols, names = zip(*mols_names)

    return filter_mols(mols, names, **kwargs)


def filter_supply(supplyfile: str, id_prop_name: Optional[str],
                  **kwargs) -> Tuple[List[str], Optional[List[str]]]:
    p_supply = Path(supplyfile)
    if p_supply.suffix == '.sdf':
        supply = Chem.SDMolSupplier(supplyfile)
    elif p_supply.suffix == '.smi':
        supply = Chem.SmilesMolSupplier(supplyfile)
    else:
        raise ValueError(
            f'input file "{supplyfile}" does not have .sdf or .smi extension')

    supply = tqdm(supply, desc='Reading in mols', unit='mol')
    mols = []
    names = None

    if id_prop_name:
        names = []
        for mol in supply:
            if mol is None:
                continue

            mols.append(mol_to_smi(mol))
            names.append(mol.GetProp(id_prop_name))
    else:
        for mol in supply:
            if mol is None:
                continue

            mols.append(mol_to_smi(mol))

    return filter_mols(mols, names, **kwargs)

def mol_to_smi(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)

def mol_from_smi(smi: str) -> Chem.Mol:
    return Chem.MolFromSmiles(smi)