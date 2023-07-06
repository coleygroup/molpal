from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from read_results_utils import parse_config, calc_true_hv, get_hvs, get_scores, pareto_profile, true_pf, top_k_smiles, get_top_k_profile, calc_front_extent
from tqdm import tqdm 
import pickle 
import matplotlib.patches as mpatches

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
    pareto_data = {label: {'all': [], 'mean': [], 'std': []} for label in ['ei', 'pi', 'nds']}
    scal_data = {label: {'all': [], 'mean': [], 'std': []} for label in ['ei', 'pi', 'greedy']}
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

