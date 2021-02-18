"""This subpackage is DEPRECATED. use the static functions inside the Vina
class instead"""
from molpal.objectives.pyscreener.docking.vina_utils.preparation import (
    prepare_receptor, prepare_from_smi, prepare_from_file
)
from molpal.objectives.pyscreener.docking.vina_utils.docking import dock_ligand, parse_score