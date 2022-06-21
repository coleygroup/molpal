import configargparse
from molpal.objectives.lookup import LookupObjective, parse_config
config = 'examples/objective/DRD2_docking.ini'
args = parse_config(config)
lookup_obj = LookupObjective(objective_config=config, minimize=True)
