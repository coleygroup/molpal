import os
from pathlib import Path
import shlex
from typing import Optional

from configargparse import ArgumentParser, ArgumentTypeError, Namespace

def positive_int(arg: str):
    value = int(arg)
    if value <= 0:
        raise ArgumentTypeError('Value must be greater than 0!')
    return value

def add_general_args(parser: ArgumentParser):
    parser.add_argument('--config', is_config_file=True,
                        help='filepath of a configuration file to use')
    parser.add_argument('--name',
                        help='the base name of the outputs')
    parser.add_argument('--mode', default='docking',
                        choices=['docking', 'md', 'dft'],
                        help='the mode in which to run pyscreener')

    parser.add_argument('--distributed', action='store_true', default=False,
                        help='whether to parallelize computation using a distributed setup')
    parser.add_argument('-nw', '-nj', '-np', '--num-workers', '--njobs',
                        type=int, default=-1,
                        metavar='N_WORKERS',
                        help='the number of workers to use. (Only used when distributed=False.)')
    
    parser.add_argument('--root', default='.',
                        help='the root directory under which to organize all program outputs')
    parser.add_argument('--copy-all', action='store_true', default=False,
                        help='whether all prepared input files and generated output files should be copied to the final output directory. By default, these files are all stored in a temporary directory.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='the level of output this program should print')

def add_preprocessing_args(parser: ArgumentParser):
    parser.add_argument('--preprocessing-options', nargs='+', default='none',
                        choices=['pdbfix', 'autobox', 'tautomers', 'desalt', 
                                 'filter'],
                        help='the preprocessing options to apply')
    parser.add_argument('--docked-ligand-file',
                        help='the name of a file containing the coordinates of a docked/bound ligand. If using Vina-type software, this file must be a PDB format file. If using Dock, do not select this preprocessing option as autoboxing occurs during input preparation.')
    parser.add_argument('--buffer', type=float, default=10.,
                        help='the amount of buffer space to add around the docked ligand when calculating the docking box.')
    parser.add_argument('--pH', type=float, default=7.,
                        help='the pH for which to calculate protonation state for protein and ligand residues')

def add_preparation_args(parser: ArgumentParser):
    parser.add_argument('--no-title-line', default=False, action='store_true',
                        help='whether there is no title line in the ligands csv file')
    parser.add_argument('--start', type=int, default=0,
                        help='the index at which start conversion')
    parser.add_argument('--nconvert', type=int,
                        help='the number of molecules to convert')
    
def add_docking_args(parser: ArgumentParser):
    parser.add_argument('--software', default='vina',
                        choices=['vina', 'smina', 'qvina', 'psovina', 'dock'],
                        help='the name of the docking program to use')
    parser.add_argument('-r', '--receptors', nargs='+',
                        help='the filenames of the receptors')
    parser.add_argument('--pdbids', nargs='+',
                        help='the PDB IDs of the crystal structure to dock against')
    parser.add_argument('-l', '--ligands', required=True, nargs='+',
                        help='the filenames containing the ligands to dock')
    parser.add_argument('--use-3d', action='store_true', default='False',
                        help='how to treat the preparation of ligands from files containing three-dimensional information. If False, use only the 2D graph of each molecule in the SDF file when preparing inputs. Faster, but will result in the loss of conformational/tautomeric information. If True, use the 3D information contained in the file when preparing an input. Slower, but will preserve conformational/tautomeric information.')
    parser.add_argument('--center', type=float, nargs=3,
                        metavar=('CENTER_X', 'CENTER_Y', 'CENTER_Z'),
                        help='the x-, y-, and z-coordinates of the center of the docking box')
    parser.add_argument('--size', type=float, nargs=3,
                        default=(10., 10., 10.),
                        metavar=('SIZE_X', 'SIZE_Y', 'SIZE_Z'),
                        help='the x-, y-, and z-dimensions of the docking box')
    
    # DOCK args
    parser.add_argument('--use-largest', action='store_true', default=False,
                        help='whether to use the largest cluster of spheres when preparing the .sph file for DOCK docking')
    parser.add_argument('--dont-enclose-spheres', action='store_true',
                        default=False,
                        help='whether to not enclose the selected spheres during DOCK docking box construction. Using this flag will manually construct the docking box using the input center and size arguments. Enclosing selected spheres is the typical way in which docking boxes are constructed for DOCK.')
                        
def add_screening_args(parser: ArgumentParser):
    parser.add_argument('-nc', '--ncpu', type=int, default=1, metavar='N_CPU',
                        help='the number of cores available to each worker process')
    parser.add_argument('--extra', type=shlex.split,
                        help='extra command line arguments to pass to screening software. E.g., "--exhaustiveness=16"')

    ### SCORING ARGS ###    
    parser.add_argument('--score-mode', default='best',
                        choices={'best', 'avg', 'boltzmann'},
                        help='The method used to calculate the score of a single docking run on a single receptor')
    parser.add_argument('--repeats', type=positive_int, default=1,
                        help='the number of times to repeat each screening run')
    parser.add_argument('--repeat-score-mode', default='best',
                        choices={'best', 'avg', 'boltzmann'},
                        help='The method used to calculate the overall score from multiple docking runs on the same receptor')
    parser.add_argument('--ensemble-score-mode', default='best',
                        choices={'best', 'avg', 'boltzmann'},
                        help='The method used to calculate the overall score from an ensemble of docking runs')

def add_postprocessing_args(parser: ArgumentParser):
    parser.add_argument('--postprocessing-options', nargs='+', default='none',
                        choices=['cluster', 'visualize'],
                        help='the postprocessing options to apply')
    parser.add_argument('--n-cluster', type=int, default=10,
                        help='the number of clusters to form')
    parser.add_argument('--viz-mode', default='text',
                        choices=['histogram', 'text'],
                        help='the type of visualization to generate. "hist" makes a histogram that is output in a pdf and "text" generates a histogram using terminal output.')

def gen_args(argv: Optional[str] = None) -> Namespace:
    parser = ArgumentParser(
        description='Automate virtual screening of compound libraries.')

    add_general_args(parser)
    add_preprocessing_args(parser)
    add_preparation_args(parser)
    add_screening_args(parser)
    add_docking_args(parser)
    add_postprocessing_args(parser)

    args = parser.parse_args(argv)

    if args.name is None:
        args.name = f'{Path(args.receptor).stem}_{Path(args.ligands).stem}'
        
    args.title_line = not args.no_title_line
    del args.no_title_line

    args.enclose_spheres = not args.dont_enclose_spheres
    del args.dont_enclose_spheres

    return args
