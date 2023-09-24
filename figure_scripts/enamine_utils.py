import numpy as np 
import pygmo as pg 
from molpal.acquirer.pareto import Pareto
from molpal.objectives.lookup import LookupDockingObjective
from read_results_utils import get_pool, calc_hv, true_pf, parse_config, get_scores, get_hvs, get_top_k_profile, pareto_profile
import pickle
from pathlib import Path
from tqdm import tqdm

def get_valid_smis_and_scores(smiles, scores, save_dir):

    scores = np.array(scores.copy(), dtype=float)
    valid_idxs = ~np.isnan(scores).any(axis=1)
    valid_scores = scores[valid_idxs]
    smiles = np.array(smiles)
    valid_smis = smiles[valid_idxs]

    with open(f'{save_dir}/valid_smiles.np', 'wb') as f:
        np.save(f, valid_smis)
    
    with open(f'{save_dir}/valid_scores.np', 'wb') as f:
        np.save(f, valid_scores)

    return valid_smis, valid_scores 


def calc_top_k(k, smiles, scores): 

    scores_unranked = scores.copy()

    this_rank = 0
    top_smis = []
    
    # pbar = tqdm(total=top_n_scored)
    while len(top_smis)/len(smiles) < k:
        this_rank = this_rank + 1

        pf = Pareto(num_objectives=scores_unranked.shape[1], compute_metrics=False)
        pf.update_front(scores_unranked)
        pf.sort_front()
        front_scores, front_num = pf.export_front()
        front_num = pf.front_num        
        
        top_smis.extend([smiles[i] for i in front_num.astype(int)])
        print(f'{100*len(top_smis)/len(smiles):0.5f}% ranked')

        if this_rank == 1: 
            front_smis = top_smis.copy()
            true_front_scores = scores_unranked[front_num].copy()

        scores_unranked[front_num] = -np.inf

    top_percent = len(top_smis)/len(smiles)
    print(f'Top {k*100}%: {100*top_percent:0.3f}')
    
    return top_smis, top_percent, front_smis, true_front_scores

def get_exhaustive_metrics(base_dir, paths_to_config, save_dir, k=0.01):

    valid_smis, valid_scores = load_scores(save_dir)

    top_smis, top_percent, front_smis, front_scores = calc_top_k(k, valid_smis, valid_scores)
    total_hv, reference_min = calc_hv(valid_scores)
    store = {
        'top_1_smis': top_smis, 
        'top_1_percent': top_percent,
        'total_hv': total_hv,
        'reference_min': reference_min, 
        'true_front_points': front_scores,
        'true_front_smis': front_smis, 
    }
    with open(save_dir/f'exhaustive_metrics_{k}.pickle','wb') as f:
        pickle.dump(store, f)

    return store 

def objs_from_paths(paths): 
    return [LookupDockingObjective(str(path)) for path in paths_to_config]

def run_metrics(base_dir, paths_to_config, save_dir, exhaustive_metrics, k): 
    run_dicts = []
    valid_smis, valid_scores = load_scores(save_dir)
    for run_folder in tqdm(sorted(list(base_dir.glob('seed*'))), desc='Reading Results'): 
        tags = parse_config(run_folder/'config.ini') 
        all_scores, all_acquired, explored_smis = get_scores_enamine(run_folder, 
                                                                     scalarize=tags['scalarize'], 
                                                                     valid_smis=valid_smis, 
                                                                     valid_scores=valid_scores)
        hv_fracs = get_hvs(all_scores, exhaustive_metrics['reference_min'])/exhaustive_metrics['total_hv']
        fronts, final_front, _, _, _, all_igd = pareto_profile(
            all_scores, 
            exhaustive_metrics['reference_min'], 
            exhaustive_metrics['true_front_points'])
        top_k_profile = get_top_k_profile(explored_smis, exhaustive_metrics['top_1_smis'])
        run_dict = {
            'all_scores': all_scores, 
            'acquired_scores': all_acquired, 
            'pf_profile': fronts,
            'final_pf': final_front,
            'scalarization': tags['scalarize'],
            'metric': tags['metric'], 
            'hv_profile': hv_fracs, 
            'seed': tags['seed'],
            'model_seed': tags['model_seed'],
            'top-k-profile': top_k_profile,
            'cluster_type': str(tags['cluster_type']),
            'overall_igd': all_igd, 
        }
        run_dicts.append(run_dict)

    with open(save_dir / f'results_{k}.pickle', 'wb') as f:
        pickle.dump(run_dicts, f)
    
    return run_dicts

def load_scores(save_dir): 
    if (Path(save_dir)/'valid_scores.np').exists() and (Path(save_dir)/'valid_smiles.np').exists(): 
        with open(save_dir/'valid_scores.np', 'rb') as f: 
            valid_scores = np.load(f)
        with open(save_dir/'valid_smiles.np', 'rb') as f: 
            valid_smis = np.load(f)
        print('loaded smiles and scores')
    else:    
        pool = get_pool(base_dir, pool_sep=",")
        objs = objs_from_paths(paths_to_config)
        data = [[obj([smi])[smi] for smi in pool.smis()] for obj in objs]
        scores = np.array(data).T
        valid_smis, valid_scores = get_valid_smis_and_scores(list(pool.smis()), scores, save_dir)
    
    return valid_smis, valid_scores

def get_scores_enamine(run_dir: Path, scalarize=False, valid_smis=None, valid_scores=None): 
    if scalarize=='True': 
        scores_dict = {smi: score for smi, score in zip(valid_smis, valid_scores)}
    
    iter_dirs = sorted((run_dir / 'chkpts').glob('iter_*'))
    its = [int(n.stem.split('_')[1]) for n in iter_dirs]
    iter_dirs = [dir for _, dir in sorted(zip(its, iter_dirs))]
    
    all_scores = []
    all_acquired = []
    explored_smis = []

    for iter in iter_dirs: 
        with open(iter/'scores.pkl', 'rb') as file: 
            explored = pickle.load(file)
        with open(iter/'new_scores.pkl', 'rb') as file: 
            acquired = pickle.load(file)

        explored_smis.append(list(explored.keys()))
        if scalarize: 
            all_scores.append(np.array([
                scores_dict[smi] for smi in explored.keys()
            ]))
            all_acquired.append(np.array([
                scores_dict[smi] for smi in acquired.keys()
            ]))
        else:
            all_scores.append(np.array(list(explored.values())))
            all_acquired.append(np.array(list(acquired.values())))
    
    return all_scores, all_acquired, explored_smis

        
if __name__=='__main__': 
    k = 0.005
    paths_to_config  =  [
        Path(f'moo_runs/enamine_runs/IGF1R_min.ini'), 
        Path(f'moo_runs/enamine_runs/EGFR_min.ini'),
        Path(f'moo_runs/enamine_runs/CYP_max.ini'),
    ]
    save_dir = Path(f'figure_scripts/enamine_IGF1R_EGFR_01')
    save_dir.mkdir(parents=False, exist_ok=True)
    base_dir = Path('results/IGF1R_EGFR_pi')
    if (save_dir / f'exhaustive_metrics_{k}.pickle').exists():
        with open(save_dir / f'exhaustive_metrics_{k}.pickle', 'rb') as f: 
            exhaustive_metrics = pickle.load(f)
    else: 
        exhaustive_metrics = get_exhaustive_metrics(base_dir, paths_to_config, save_dir, k)

    runs = run_metrics(base_dir, paths_to_config, save_dir, exhaustive_metrics, k)
