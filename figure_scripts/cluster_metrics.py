from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
from read_results_utils import  calc_true_hv, true_pf
from organize_results import create_run_dicts
from plot_utils import set_size, set_style, it_colors, shapes, cluster_colors, cluster_labels
import pickle 
import seaborn as sns 
import pandas as pd 

set_style()
filetype='.pdf'
target = 'IGF1R'
offt = 'CYP'

base_dir = Path('results') / f'{target}_clustering'
save_dir = Path(f'figure_scripts/cluster_figs')
save_dir.mkdir(parents=False, exist_ok=True)
obj_configs =  [Path(f'moo_runs/objective/{target}_min.ini'), Path(f'moo_runs/objective/{offt}_max.ini')]
pool_sep = "\t"

k = 0.01

if not (save_dir/'results.pickle').exists(): 
    run_dicts, true_front = create_run_dicts(base_dir, obj_configs, pool_sep=pool_sep, k=k)
    with open(save_dir/'results.pickle', 'wb') as file: 
        pickle.dump(run_dicts, file)
    with open(save_dir/'true_front.pickle', 'wb') as file: 
        pickle.dump(true_front, file)
    print('done')
elif not (save_dir/'true_front.pickle').exists(): 
    _, reference_min, objs, true_scores = calc_true_hv(obj_configs, base_dir, pool_sep)
    true_front = true_pf(true_scores, reference_min)
    with open(save_dir/'true_front.pickle', 'wb') as file: 
        pickle.dump(true_front, file)
    with open(save_dir/'results.pickle', 'rb') as file: 
        run_dicts = pickle.load(file)
else: 
    with open(save_dir/'results.pickle', 'rb') as file: 
        run_dicts = pickle.load(file)
    with open(save_dir/'true_front.pickle', 'rb') as file: 
        true_front = pickle.load(file)


def extract_cluster_data(run_dicts, key: str): 
    data = {label: {'all': [], 'mean': [], 'std': []} for label in cluster_labels.keys()}
    
    for run_dict in run_dicts:
        if run_dict['metric'] == 'pi': 
            data[run_dict['cluster_type']]['all'].append(run_dict[key])
    
    for entry in data.values(): 
        entry['mean'] = np.array(entry['all']).mean(axis=0) 
        entry['std'] = np.array(entry['all']).std(axis=0)
    
    return data 

def subfig_top_k(run_dicts): 
    data = extract_cluster_data(run_dicts, 'top-k-profile')

    x = range(len(data['None']['mean']))
    fig, ax = plt.subplots(1,1)

    for c_type, entry in data.items(): 
        ax.plot(x, entry['mean'], label=cluster_labels[c_type], color=cluster_colors[c_type],)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = cluster_colors[c_type], alpha=0.3
            )
    
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    ylim = ax.get_ylim()
    ax.set_ylabel('Fraction of top ~1%')
    ax.locator_params(axis='y', nbins=5)
    ax.set_xticks(x)
    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'top_k_1{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def plot_end_top_k(run_dicts): 

    data = extract_cluster_data(run_dicts, 'top-k-profile')
    
    for entry in data.values(): 
        entry['mean'] = entry['mean'][-1]
        entry['std'] = entry['std'][-1]

    w = 0.25
    x = np.array(range(4))/2

    fig, ax = plt.subplots(1,1)

    for i, c_type in enumerate(cluster_colors.keys()): 
        ax.bar(x[i], data[c_type]['mean'], width=w, color=cluster_colors[c_type], align='center')
        ax.errorbar(x[i], data[c_type]['mean'], yerr=data[c_type]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)

    ax.set_xticks(x, cluster_labels.values())
    ax.locator_params(axis='y', nbins=5)
    ax.set_ylabel('Fraction of top ~1%')
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1]+0.1])

    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'end_top_k{filetype}',bbox_inches='tight', dpi=200, transparent=True)


def final_pf(run_dicts, true_front, seed=47, model_seed=29, c_type='None'):
    data = {label: [] for label in ['fps', 'objs', 'both', 'None']}

    data = [run for run in run_dicts if run['seed'] == str(seed) and run['model_seed'] == str(model_seed) and run['cluster_type']==c_type][0]
    front = data['final_pf']

    fig, ax = plt.subplots(1,1)
    ax.plot(true_front[:,0], true_front[:,1], marker=shapes[-1], color=it_colors[-1], linewidth=0.5)

    ax.plot(front[:,0], front[:,1], marker='v', color=cluster_colors[c_type], linewidth=0.5, label=cluster_labels[c_type])
    
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper right', labelspacing=0.3)
    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'end_front_{c_type}{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def fraction_true_front(run_dicts, true_front): 
    
    for entry in run_dicts: 
        pf = entry['final_pf']
        frac = sum([point.tolist() in true_front.tolist() for point in pf])/len(true_front)
        entry['fraction_first_rank'] = frac

    data = extract_cluster_data(run_dicts, 'fraction_first_rank')

    w = 0.25
    x = np.array(range(4))/2

    fig, ax = plt.subplots(1,1)

    for i, c_type in enumerate(cluster_labels.keys()): 
        ax.bar(x[i], data[c_type]['mean'], width=w, color=cluster_colors[c_type], align='center')
        ax.errorbar(x[i], data[c_type]['mean'], yerr=data[c_type]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)

    ax.set_xticks(x, cluster_labels.values())
    ax.locator_params(axis='x', nbins=5)
    ax.set_ylabel('Fraction of front')

    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'fraction_non_dominated{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def scatter_acquired(run_dicts, model_seed, seed): 
    c_types = []
    scores_all = []

    for c_type in cluster_labels.keys(): 
        scored = [d['all_scores'] for d in run_dicts if d['model_seed']==str(model_seed) and d['seed']==str(seed) and d['cluster_type']==c_type and d['metric']=='pi']
        scores = scored[0][-1]
        c_types.append([f'{c_type}']*len(scores))
        scores_all.append(scores)

    c_types = [i for sublist in c_types for i in sublist ]
    scores_all = np.concatenate(scores_all, axis=0)

    dd = {'Cluster Type': c_types, f'-{target}': scores_all[:,0], f'{offt}': scores_all[:,1]}
    df = pd.DataFrame(dd)

    fig, ax = plt.subplots(1,1)
    # ax.scatter(true_front[:,0], true_front[:,1], s=1, alpha=0.1, color='lightgray', linewidth=0, label='')
    sns.jointplot(data=df, x=f'-{target}', y=f'{offt}', hue='Cluster Type', linewidth=0, s=2)

    # set_size(1.5, 1.5, ax=ax)
    plt.grid(False)
    plt.savefig(save_dir/f'scatter_acquired_{c_type}{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def igd(run_dicts): 
    
    data = extract_cluster_data(run_dicts, 'overall_igd')
    
    w = 0.25
    x = np.array(range(4))/2

    fig, ax = plt.subplots(1,1)

    for i, c_type in enumerate(cluster_labels.keys()): 
        ax.bar(x[i], data[c_type]['mean'], width=w, color=cluster_colors[c_type], align='center')
        ax.errorbar(x[i], data[c_type]['mean'], yerr=data[c_type]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)

    ax.set_xticks(x, cluster_labels.values())
    ax.locator_params(axis='x', nbins=5)
    ax.set_ylabel('IGD')

    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'IGD{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def hypervolume(run_dicts): 
    data = extract_cluster_data(run_dicts, 'hv_profile')

    x = range(len(data['None']['mean']))
    fig, ax = plt.subplots(1,1)

    for c_type, entry in data.items(): 
        ax.plot(x, entry['mean'], label=cluster_labels[c_type], color=cluster_colors[c_type],)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = cluster_colors[c_type], alpha=0.3
            )
    
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    ylim = ax.get_ylim()
    ax.set_ylabel('Fraction of top ~1%')
    ax.locator_params(axis='y', nbins=5)
    ax.set_xticks(x)
    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'hv{filetype}',bbox_inches='tight', dpi=200, transparent=True)   


if __name__=='__main__':
    #subfig_top_k(run_dicts)
    plot_end_top_k(run_dicts)
    # for c_type in ['None', 'objs', 'fps', 'both']:
    #     final_pf(run_dicts, true_front, c_type=c_type)
    # fraction_true_front(run_dicts, true_front)
    # scatter_acquired(run_dicts, 29, 47)
    # igd(run_dicts)
    # hypervolume(run_dicts)

