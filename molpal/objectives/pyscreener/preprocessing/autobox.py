"""This module contains functions for ligand autoboxing in docking
simulations"""

from itertools import chain, takewhile
from typing import List, Optional, Tuple

def autobox(receptors: Optional[List[str]] = None,
            residues: Optional[List[int]] = None,
            docked_ligand_file: Optional[str] = None,
            buffer: int = 10, **kwargs) -> Tuple[Tuple, Tuple]:
    if residues:
        center, size = residues(receptors[0], residues)
        print('Autoboxing from residues with', end=' ')
    else:
        # allow user to only specify one receptor file
        docked_ligand_file = docked_ligand_file or receptors[0]
        center, size = docked_ligand(docked_ligand_file, buffer)
        print('Autoboxing from docked ligand with', end=' ')

    s_center = f'({center[0]:0.1f}, {center[1]:0.1f}, {center[2]:0.1f})'
    s_size = f'({size[0]:0.1f}, {size[1]:0.1f}, {size[2]:0.1f})'
    print(f'center={s_center} and size={s_size}')

    return center, size

def residues(pdbfile: str, residues: List[int]) -> Tuple[Tuple, Tuple]:
    """Generate a ligand autobox from a list of protein residues

    The ligand autobox is the minimum bounding box of the alpha carbons of the
    listed residues

    Parameters
    ----------
    pdbfile : str
        a PDB-format file containing the protein of interest
    residues: List[int]
        the residue number corresponding to the residues from which to
        calculate the autobox

    Returns
    -------
    center: Tuple[float, float, float]
        the x-, y-, and z-coordinates of the ligand autobox center
    size: Tuple[float, float, float]
        the x-, y-, and z-radii of the ligand autobox
    """
    residues = set(residues)
    with open(pdbfile) as fid:
        residue_coords = []
        for line in fid:    # advance to the atom information lines
            if 'ATOM' in line:
                break
        fid = chain([line], fid)    # prepend the first line to the generator
        for line in fid:
            record, _, a_type, _, _, res_num, x, y, z, _, _, _ = line.split()
            if 'ATOM' != record:
                break

            if res_num in residues and a_type == 'CA':
                residue_coords.append((float(x), float(y), float(z)))
    
    return minimum_bounding_box(residue_coords)

def docked_ligand(docked_ligand_file: str,
                  buffer: int = 10) -> Tuple[Tuple, Tuple]:
    """Generate a ligand autobox from a PDB file containing a docked ligand

    The ligand autobox is the minimum bounding box of the docked ligand with
    an equal buffer in each dimension. The PDB file should be either just the
    protein with a single docked ligand or a PDB file containing just the
    docked ligand.

    Parameters
    ----------
    docked_ligand_file : str
        a PDB-format file containing the coordinates of a docked ligand
    buffer : int (Default = 10)
        the buffer to add around the ligand autobox, in Angstroms

    Returns
    -------
    center: Tuple[float, float, float]
        the x-, y-, and z-coordinates of the ligand autobox center
    size: Tuple[float, float, float]
        the x-, y-, and z-radii of the ligand autobox
    """
    with open(docked_ligand_file) as fid:
        for line in fid:
            if 'HETATM' in line:
                break
        fid = chain([line], fid)    # prepend the first line to the generator
        ligand_atom_coords = [
            parse_xyz(line)
            for line in takewhile(lambda line: 'HETATM' in line, fid)
        ]

    return minimum_bounding_box(ligand_atom_coords, buffer)

def parse_xyz(line: str) -> Tuple[float, float, float]:
    return tuple(map(float, line.split()[5:8]))

def minimum_bounding_box(coords: List[Tuple[float, float, float]], 
                         buffer: float = 10.) -> Tuple[Tuple, Tuple]:
    """Calculate the minimum bounding box for a list of coordinates

    Parameters
    ----------
    coords : List[Tuple[float, float, float]]
        a list of tuples corresponding to x-, y-, and z-coordinates
    buffer : float (Default = 10.)
        the amount of buffer to add to the minimum bounding box

    Returns
    -------
    center: Tuple[float, float, float]
        the x-, y-, and z-coordinates of the center of the minimum bounding box
    size: Tuple[float, float, float]
        the x-, y-, and z-radii of the minimum bounding box
    """
    xs, ys, zs = zip(*coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2

    size_x = (max_x - center_x) + buffer
    size_y = (max_y - center_y) + buffer
    size_z = (max_z - center_z) + buffer

    center = center_x, center_y, center_z
    size = size_x, size_y, size_z

    return center, size