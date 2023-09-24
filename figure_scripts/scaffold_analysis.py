from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import EngFormatter

from read_results_utils import  calc_true_hv, true_pf
from organize_results import scaffold_storage
from plot_utils import set_size, set_style, it_colors, shapes, cluster_colors, cluster_labels
import pickle 
import seaborn as sns 
import pandas as pd 

set_style()
filetype='.pdf'
target = 'IGF1R'
offt = 'CYP'

base_dir = Path('results') / f'{target}_clustering'
save_dir = Path(f'figure_scripts/scaffold_analysis')
save_dir.mkdir(parents=False, exist_ok=True)
obj_configs =  [Path(f'moo_runs/objective/{target}_min.ini'), Path(f'moo_runs/objective/{offt}_max.ini')]
pool_sep = "\t"

if not (save_dir/'results.pickle').exists(): 
    run_dicts = scaffold_storage(base_dir, obj_configs, pool_sep=pool_sep)
    with open(save_dir/'results.pickle', 'wb') as file: 
        pickle.dump(run_dicts, file)
else: 
    with open(save_dir/'results.pickle', 'rb') as file: 
        run_dicts = pickle.load(file)

def extract_cluster_data(run_dicts, key: str): 
    data = {label: {'all': [], 'mean': [], 'std': []} for label in cluster_labels.keys()}
    
    for run_dict in run_dicts:
        if run_dict['metric'] == 'pi': 
            data[run_dict['cluster_type']]['all'].append(run_dict[key])
    
    for entry in data.values(): 
        entry['mean'] = np.array(entry['all']).mean(axis=0) 
        entry['std'] = np.array(entry['all']).std(axis=0)
    
    return data 

def scaffold_number_profile(run_dicts): 
    data = extract_cluster_data(run_dicts, 'n_scaffold_profile')

    x = range(len(data['None']['mean']))
    fig, ax = plt.subplots(1,1)

    for c_type, entry in data.items(): 
        ax.plot(x, entry['mean']/1000, label=cluster_labels[c_type], color=cluster_colors[c_type],)
        ax.fill_between(x, (entry['mean'] - entry['std'])/1000, (entry['mean'] + entry['std'])/1000, 
                facecolor = cluster_colors[c_type], alpha=0.3
            )
    
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    ylim = ax.get_ylim()
    ax.set_ylabel('Thousands of scaffolds')
    ax.locator_params(axis='y', nbins=7)
    ax.set_xticks(x)
    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'num_scaffold{filetype}',bbox_inches='tight', dpi=200, transparent=False)

def generic_scaffold_number_profile(run_dicts): 
    data = extract_cluster_data(run_dicts, 'n_generic_scaffold_profile')

    x = range(len(data['None']['mean']))
    fig, ax = plt.subplots(1,1)
    formatter0 = EngFormatter(sep="")
    ax.yaxis.set_major_formatter(formatter0)

    for c_type, entry in data.items(): 
        ax.plot(x, entry['mean'], label=cluster_labels[c_type], color=cluster_colors[c_type],)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = cluster_colors[c_type], alpha=0.3
            )
    
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    ylim = ax.get_ylim()
    ax.set_ylabel('Number of scaffolds')
    ax.locator_params(axis='y', nbins=5)
    ax.set_yticks([2000, 4000, 6000, 8000, 10000])
    ax.set_xticks(x)
    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'num_generic_scaffold{filetype}',bbox_inches='tight', dpi=200, transparent=False)



if __name__=='__main__':
    # scaffold_number_profile(run_dicts)
    generic_scaffold_number_profile(run_dicts)