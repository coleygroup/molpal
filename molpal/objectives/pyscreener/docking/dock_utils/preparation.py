try:
    import importlib.resources as resources
except ModuleNotFoundError:
    import importlib_resources as resources
from itertools import takewhile
import os
from pathlib import Path
import shutil
import subprocess as sp
import sys
from typing import Dict, Iterable, Optional, Tuple

with resources.path('pyscreener.docking.dock_utils', '.') as p_module:
    PREP_REC = p_module / 'scripts' / 'prep_rec.py'
    WRITE_DMS = p_module / 'scripts' / 'write_dms.py'

DOCK6 = Path(os.environ['DOCK6'])
DOCK6_BIN = DOCK6 / 'bin'
DOCK6_PARAMS = DOCK6 / 'parameters'

DMS = DOCK6_BIN / 'dms'
SPHGEN = DOCK6_BIN / 'sphgen_cpp'
SPHERE_SELECTOR = DOCK6_BIN / 'sphere_selector'
SHOWBOX = DOCK6_BIN / 'showbox'
GRID = DOCK6_BIN / 'grid'

VDW_DEFN_FILE = DOCK6_PARAMS / 'vdw_AMBER_parm99.defn'

def prepare_from_smi(smi: str, name: str = 'ligand',
                     path: str = '.', **kwargs) -> Optional[Tuple]:
    """Prepare an input ligand file from the ligand's SMILES string

    Parameters
    ----------
    smi : str
        the SMILES string of the ligand
    name : Optional[str] (Default = None)
        the name of the ligand.
    path : str (Default = '.')
        the path under which the output PDBQT file should be written
    **kwargs
        additional and unused keyword arguments

    Returns
    -------
    Optional[Tuple]
        a tuple of the SMILES string and the corresponding prepared input file.
        None if preparation failed for any reason
    """
    path = Path(path)
    if not path.is_dir():
        path.mkdir()
    
    mol2 = str(path / f'{name}.mol2')

    argv = ['obabel', f'-:{smi}', '-omol2', '-O', mol2,
            '-h', '--gen3d', '--partialcharge', 'gasteiger']
    ret = sp.run(argv, check=False, stderr=sp.PIPE)

    try:
        ret.check_returncode()
        return smi, mol2
    except sp.SubprocessError:
        return None

def prepare_from_file(filename: str, use_3d: bool = False,
                      name: Optional[str] = None, path: str = '.', 
                      **kwargs) -> Optional[Tuple]:
    """Convert a single ligand to the appropriate input format

    Parameters
    ----------
    filename : str
        the name of the file containing the ligand
    use_3d : bool (Default = False)
        whether to use the 3D information in the input file (if possible)
    prepare_from_smi: Callable[..., Tuple[str, str]]
        a function that prepares an input ligand file from a SMILES string
    name : Optional[str] (Default = None)
        the name of the ligand. If None, use the stem of the input file
    path : str (Default = '.')
        the path under which the output .pdbqt file should be written
    **kwargs
        additional and unused keyword arguments

    Returns
    -------
    Optional[List[Tuple]]
        a tuple of the SMILES string the prepared input file corresponding
        to the molecule contained in filename
    """
    name = name or Path(filename).stem

    ret = sp.run(['obabel', filename, '-osmi'], stdout=sp.PIPE, check=True)
    lines = ret.stdout.decode('utf-8').splitlines()
    smis = [line.split()[0] for line in lines]

    if not use_3d:
        ligands = [prepare_from_smi(smi, f'{name}_{i}', path) 
                   for i, smi in enumerate(smis)]
        return [lig for lig in ligands if lig]
    
    path = Path(path)
    if not path.is_dir():
        path.mkdir()

    mol2 = f'{path}/{name}_.mol2'
    argv = ['obabel', filename, '-omol2', '-O', mol2, '-m',
            '-h', '--partialcharge', 'gasteiger']

    ret = sp.run(argv, check=False, stderr=sp.PIPE)
    try:
        ret.check_returncode()
    except sp.SubprocessError:
        return None

    stderr = ret.stderr.decode('utf-8')
    for line in stderr.splitlines():
        if 'converted' not in line:
            continue
        n_mols = int(line.split()[0])

    mol2s = [f'{path}/{name}_{i}.mol2' for i in range(1, n_mols)]

    return list(zip(smis, mol2s))

def prepare_receptor(receptor: str, probe_radius: float = 1.4,
                     steric_clash_dist: float = 0.0,
                     min_radius: float = 1.4, max_radius: float = 4.0,
                     center: Optional[Tuple[float, float, float]] = None,
                     size: Tuple[float, float, float] = (20., 20., 20.), 
                     docked_ligand_file: Optional[str] = None,
                     use_largest: bool = False, buffer: float = 10.,
                     enclose_spheres: bool = True,
                     path: str = '.') -> Optional[Tuple[str, str]]:
    """Prepare the DOCK input files corresponding to the given receptor

    Parameters
    ----------
    receptor : str
        the filepath of a file containing a receptor. Must be in a file that
        is readable by Chimera
    center : Tuple[float, float, float]
        the x-, y-, and z-coordinates of the center of the docking box
    size : Tuple[float, float, float] (Default = (20, 20, 20))
        the x-, y-, and z-radii of the docking box
    docked_ligand_file : Optional[str] (Default = None)
        the filepath of a file containing the coordinates of a docked ligand
    use_largest : bool (Default = False)
        whether to use the largest cluster of spheres when selecting spheres
    buffer : float (Default = 10.)
        the amount of buffer space to be added around the docked ligand when
        selecting spheres and when constructing the docking box if 
        enclose_spheres is True
    enclose_spheres : bool (Default = True)
        whether to calculate the docking box by enclosing the selected spheres
        or to use an input center and radii

    Returns
    -------
    sph_grid : Optional[Tuple[str, str]]
        A tuple of strings with the first entry being the filepath of the file 
        containing the selected spheres and the second being entry the prefix 
        of all prepared grid files. None if receptor preparation fails at any 
        point
    """
    rec_mol2 = prepare_mol2(receptor, path)
    rec_pdb = prepare_pdb(receptor, path)
    if rec_mol2 is None or rec_pdb is None:
        return None

    rec_dms = prepare_dms(rec_pdb, probe_radius, path)
    if rec_dms is None:
        return None

    rec_sph = prepare_sph(rec_dms, steric_clash_dist,
                          min_radius, max_radius, path)
    if rec_sph is None:
        return None

    rec_sph = select_spheres(
        rec_sph, center, size,
        docked_ligand_file, use_largest, buffer, path
    )

    rec_box = prepare_box(rec_sph, center, size, enclose_spheres, buffer, path)
    if rec_box is None:
        return None

    grid_prefix = prepare_grid(rec_mol2, rec_box, path)
    if grid_prefix is None:
        return None

    return rec_sph, grid_prefix

def prepare_mol2(receptor: str, path: str = '.') -> Optional[str]:
    """Prepare a receptor mol2 file from its input file

    Parameter
    ---------
    receptor : str
        the filename of a file containing the receptor

    Returns
    -------
    receptor_mol2 : Optional[str]
        the filename of the prepared mol2 file
    """
    p_rec = Path(receptor)
    p_rec_mol2 = Path(path) / f'{p_rec.stem}_withH.mol2'
    # (p_rec.with_name(f'{p_rec.stem}_withH.mol2'))
    args = ['chimera', '--nogui', '--script',
            f'{PREP_REC} {receptor} {p_rec_mol2}']

    ret = sp.run(args, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        ret.check_returncode()
    except sp.SubprocessError:
        print(f'ERROR: failed to convert receptor: "{receptor}"')
        if ret.stderr:
            print(f'Message: {ret.stderr.decode("utf-8")}', file=sys.stderr)
        return None

    return str(p_rec_mol2)

def prepare_pdb(receptor: str, path: str = '.') -> Optional[str]:
    """Prepare a receptor PDB file for usage in DOCK runs

    Parameter
    ---------
    receptor : str
        the filename of a file containing the receptor

    Returns
    -------
    receptor_mol2 : Optional[str]
        the filename of the prepared pdb file
    """
    p_rec = Path(receptor)
    rec_pdb = str(Path(path) / f'DOCK_{p_rec.stem}.pdb')
    # rec_pdb = str(p_rec.with_name(f'DOCK_{p_rec.stem}.pdb'))
    args = ['obabel', receptor, '-opdb', '-O', rec_pdb]

    ret = sp.run(args, stderr=sp.PIPE)
    try:
        ret.check_returncode()
    except sp.SubprocessError:
        print(f'ERROR: failed to convert receptor: "{receptor}"')
        if ret.stderr:
            print(f'Message: {ret.stderr.decode("utf-8")}', file=sys.stderr)
        return None
    
    return rec_pdb
    
def prepare_dms(rec_pdb: str, probe_radius: float = 1.4,
                path: str = '.') -> Optional[str]:
    # p_rec_pdb = Path(rec_pdb)
    # rec_dms = str(Path(rec_pdb).with_suffix('.dms'))
    p_rec_dms = Path(path) / f'{Path(rec_pdb).stem}.dms'
    argv = ['chimera', '--nogui', '--script',
            f'{WRITE_DMS} {rec_pdb} {probe_radius} {str(p_rec_dms)}']

    ret = sp.run(argv, stdout=sp.PIPE)
    try:
        ret.check_returncode()
    except sp.SubprocessError:
        print(f'ERROR: failed to generate surface from "{rec_pdb}"',
              file=sys.stderr)
        if ret.stderr:
            print(f'Message: {ret.stderr.decode("utf-8")}', file=sys.stderr)
        # return None

    return str(p_rec_dms)

def prepare_sph(rec_dms: str, steric_clash_dist: float = 0.0,
                min_radius: float = 1.4, max_radius: float = 4.0,
                path: str = '.') -> Optional[str]:
    # sph_file = str(Path(rec_dms).with_suffix('.sph'))
    sph_file = str(Path(path) / f'{Path(rec_dms).stem}.sph')
    argv = [SPHGEN, '-i', rec_dms, '-o', sph_file, 
            '-s', 'R', 'd', 'X', '-l', str(steric_clash_dist),
            'm', str(min_radius), '-x', str(max_radius)]

    ret = sp.run(argv, stdout=sp.PIPE)
    try:
        ret.check_returncode()
    except sp.SubprocessError:
        print(f'ERROR: failed to generate spheres for "{rec_dms}"',
              file=sys.stderr)
        if ret.stderr:
            print(f'Message: {ret.stderr.decode("utf-8")}', file=sys.stderr)
        return None
    
    return sph_file

def select_spheres(sph_file: str, 
                   center: Tuple[float, float, float],
                   size: Tuple[float, float, float],
                   docked_ligand_file: Optional[str] = None,
                   use_largest: bool = False,
                   buffer: float = 10.0, path: str = '.') -> Optional[str]:
    p_sph = Path(sph_file)
    selected_sph = str(Path(path) / f'{p_sph.stem}_selected_spheres.sph')
    # selected_sph = str(p_sph.parent / f'{p_sph.stem}_selected{p_sph.suffix}')

    if docked_ligand_file:
        argv = [SPHERE_SELECTOR, sph_file, docked_ligand_file, buffer]
        sp.run(argv, check=True)
        # sphere_selector always outputs this filename
        shutil.move('selected_spheres.sph', selected_sph)
        return selected_sph

    def inside_docking_box(line):
        """Are the coordinates contained in the line inside the docking box?"""
        try:
            tokens = line.split()
            xyz = list(map(float, tokens[1:4]))
        except ValueError:
            return False

        for i, coord in enumerate(xyz):
            if not center[i]-size[i] <= coord <= center[i]+size[i]:
                return False
        return True

    with open(sph_file, 'r') as fin, open(selected_sph, 'w') as fout:
        if use_largest:
            fout.write(f'DOCK spheres largest cluster\n')
            for line in takewhile(lambda line: 'cluster' not in line, fin):
                continue
            lines = list(takewhile(lambda line: 'cluster' not in line, fin))
        else:
            fout.write(f'DOCK spheres within radii {size} of {center}\n')
            lines = [line for line in fin if inside_docking_box(line)]

        fout.write(f'cluster     1 number of spheres in cluster {len(lines)}\n')
        fout.writelines(lines)

    return selected_sph

def prepare_box(sph_file: str,
                center: Tuple[float, float, float],
                size: Tuple[float, float, float],
                enclose_spheres: bool = True,
                buffer: float = 10.0, path: str = '.') -> Optional[str]:
    p_sph = Path(sph_file)
    shutil.copyfile(sph_file, 'tmp_spheres.sph')
    # p_box = p_sph.with_name(f'{p_sph.stem}_box.pdb')
    box_file = str(Path(path) / f'{p_sph.stem}_box.pdb')
    # box_file = str(p_box)

    if enclose_spheres:
        showbox_input = f'Y\n{buffer}\ntmp_spheres.sph\n1\n'
    else:
        x, y, z = center
        r_x, r_y, r_z = size
        showbox_input = f'N\nU\n{x} {y} {z}\n{r_x} {r_y} {r_z}\n'
    showbox_input += 'tmp_box.pdb\n'

    # with open('box.in', 'w') as fid:
    #     if enclose_spheres:
    #         fid.write('Y\n')
    #         fid.write(f'{buffer}\n')
    #         fid.write(f'{sph_file}\n')
    #         fid.write(f'1\n')
    #     else:
    #         fid.write('N\n')
    #         fid.write('U\n')
    #         fid.write(f'[{center[0]} {center[1]} {center[2]}]\n')
    #         fid.write(f'[{size[0]} {size[1]} {size[2]}]\n')
    #     fid.write(f'{box_file}\n')
    
    ret = sp.run([SHOWBOX], input=showbox_input, universal_newlines=True,
                 stdout=sp.PIPE)
    try:
        ret.check_returncode()
    except sp.SubprocessError:
        print(f'ERROR: failed to generate box corresponding to "{sph_file}"',
              file=sys.stderr)
        if ret.stderr:
            print(f'Message: {ret.stderr.decode("utf-8")}', file=sys.stderr)
        return None

    os.unlink('tmp_spheres.sph')
    shutil.move('tmp_box.pdb', box_file)
    return box_file

def prepare_grid(rec_mol2: str, box_file: str,
                 path: str = '.') -> Optional[str]:
    p_rec = Path(rec_mol2)
    p_grid_prefix = Path(path) / f'{p_rec.stem}_grid'

    shutil.copy(box_file, 'tmp_box.pdb')
    with open('grid.in', 'w') as fid:
        fid.write('compute_grids yes\n')
        fid.write('grid_spacing 0.4\n')
        fid.write('output_molecule no\n')
        fid.write('contact_score no\n')
        fid.write('energy_score yes\n')
        fid.write('energy_cutoff_distance 9999\n')
        fid.write('atom_model a\n')
        fid.write('attractive_exponent 6\n')
        fid.write('repulsive_exponent 12\n')
        fid.write('distance_dielectric yes\n')
        fid.write('dielectric_factor 4.0\n')
        fid.write('bump_filter yes\n')
        fid.write('bump_overlap 0.75\n')
        fid.write(f'receptor_file {rec_mol2}\n')
        fid.write('box_file tmp_box.pdb\n')
        fid.write(f'vdw_definition_file {VDW_DEFN_FILE}\n')
        fid.write(f'score_grid_prefix  {p_grid_prefix}\n')
    
    ret = sp.run([GRID, '-i', 'grid.in', '-o', 'gridinfo.out'], stdout=sp.PIPE)
    try:
        ret.check_returncode()
    except sp.SubprocessError:
        print(f'ERROR: failed to generate grid from {rec_mol2}',
              file=sys.stderr)
        if ret.stderr:
            print(f'Message: {ret.stderr.decode("utf-8")}', file=sys.stderr)
        return None

    os.unlink('tmp_box.pdb')
    return str(p_grid_prefix)
