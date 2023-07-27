import numpy as np 
from read_results_utils import parse_config, calc_true_hv, get_hvs, get_scores, pareto_profile, true_pf, top_k_smiles, get_top_k_profile, calc_front_extent
from tqdm import tqdm 
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from tqdm import tqdm

def scaffold_storage(base_dir, obj_configs, pool_sep=','):
    _, _, objs, _, all_smiles = calc_true_hv(obj_configs, base_dir, pool_sep)
    run_dicts = []
    scaffold_dict, generic_dict = get_scaffold_map(all_smiles)
    for run_folder in tqdm(list(base_dir.glob('seed*')), desc='Reading Results'): 
        tags = parse_config(run_folder/'config.ini') 
        _, _, explored_smis = get_scores(run_folder, objs)
        scaffolds, number = get_scaffold_profile(explored_smis, scaffold_dict)
        generic_scaffs, number_generic = get_scaffold_profile(explored_smis, generic_dict)
        run_dict = {
            'n_scaffold_profile': number,
            'acquired_scaffolds': scaffolds,
            'explored_smis': explored_smis,
            'n_generic_scaffold_profile': number_generic,
            'acquired_generic_scaffolds': generic_scaffs,            
            'scalarization': tags['scalarize'],
            'metric': tags['metric'], 
            'seed': tags['seed'],
            'model_seed': tags['model_seed'],
            'cluster_type': str(tags['cluster_type']),
        }
        run_dicts.append(run_dict)
    
    return run_dicts

def get_scaffold_profile(explored_smis, scaffold_dict):
    scaffolds = []
    number = []
    for smis in explored_smis: 
        iter_scaff = set([scaffold_dict[smi] for smi in smis])
        scaffolds.append(iter_scaff) 
        number.append(len(iter_scaff))
    return scaffolds, number 

def get_scaffold_map(smis): 
    scaffold_dict = {smi: MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi) for smi in tqdm(smis, desc='getting scaffolds')}
    generic_scaffold_dict = {smi: Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(Chem.MolFromSmiles(scaff)))
                             for smi, scaff in tqdm(scaffold_dict.items(), 'making scaffolds generic')}
    return scaffold_dict, generic_scaffold_dict

def create_run_dicts(base_dir, obj_configs, k=0.01, pool_sep=','):
    runs = []
    run_folders = base_dir.glob('seed*')
    true_hv, reference_min, objs, true_scores, all_smiles = calc_true_hv(obj_configs, base_dir, pool_sep)
    top_k_smis, top_percent = top_k_smiles(k, all_smiles, true_scores)
    true_front = true_pf(true_scores, reference_min)
    run_dicts = []
    for run_folder in tqdm(list(base_dir.glob('seed*')), desc='Reading Results'): 
        tags = parse_config(run_folder/'config.ini') 
        all_scores, all_acquired, explored_smis = get_scores(run_folder, objs)
        hv_fracs = get_hvs(all_scores, reference_min, objs)/true_hv
        fronts, final_front, final_gd, all_gd, final_igd, all_igd = pareto_profile(all_scores, reference_min, true_front)
        top_k_profile = get_top_k_profile(explored_smis, top_k_smis)
        run_dict = {
            'all_scores': all_scores, 
            'acquired_scores': all_acquired, 
            'pf_profile': fronts,
            'final_pf': final_front,
            'scalarization': tags['scalarize'],
            'metric': tags['metric'], 
            'hv_profile': hv_fracs, 
            'final_gd': final_gd,
            'seed': tags['seed'],
            'model_seed': tags['model_seed'],
            'top-k-profile': top_k_profile,
            'overall_gd': all_gd, 
            'final_extent': calc_front_extent(final_front),
            'cluster_type': str(tags['cluster_type']),
            'overall_igd': all_igd, 
            'final_igd': final_igd,
        }
        run_dicts.append(run_dict)
    
    return run_dicts, true_front

def extract_data(run_dicts, key: str):
    pareto_data = {label: {'all': [], 'mean': [], 'std': []} for label in ['nds', 'ei', 'pi', ]}
    scal_data = {label: {'all': [], 'mean': [], 'std': []} for label in ['greedy', 'ei', 'pi', ]}
    rand_data = {'all': [], 'mean': [], 'std': []}

    for run_dict in run_dicts: 
        if run_dict['metric'] == 'random': 
            rand_data['all'].append(run_dict[key])
            continue
        data = scal_data if run_dict['scalarization'] else pareto_data
        data[run_dict['metric']]['all'].append(run_dict[key])
    
    for entry in [*pareto_data.values(), *scal_data.values(), rand_data]: 
        entry['mean'] = np.array(entry['all']).mean(axis=0) 
        entry['std'] = np.array(entry['all']).std(axis=0)
    
    return pareto_data, scal_data, rand_data

