from pathlib import Path 
import pickle 
import numpy as np 
import matplotlib.pyplot as plt
from plot_utils import acq_labels, method_colors, set_size, set_style, shapes, cluster_colors
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import seaborn as sns 


set_style()
filetype='.pdf'
k = 0.001
batch_size = 0.01
transp = False
save_dir = Path(f'figure_scripts/enamine_IGF1R_EGFR_01')
save_dir.mkdir(parents=False, exist_ok=True)
base_dir = Path('results/IGF1R_EGFR_pi')
paths_to_config  =  [
    Path(f'moo_runs/enamine_runs/IGF1R_min.ini'), 
    Path(f'moo_runs/enamine_runs/EGFR_min.ini'),
    Path(f'moo_runs/enamine_runs/CYP_max.ini'),
]

with open(save_dir/f'results_{k}.pickle','rb') as f: 
    run_dicts = pickle.load(f)
with open(save_dir/f'exhaustive_metrics_{k}.pickle','rb') as f:
    exhaustive_metrics = pickle.load(f)

def extract_data(run_dicts, key): 
    data = {metric: {'all': [], 'mean': [], 'std': [], 'model_seeds': [], 'seeds': []} for metric in ['pi','random']}

    for run_dict in run_dicts: 
        metric = run_dict['metric']
        data[metric]['all'].append(run_dict[key])
        data[metric]['seeds'].append(run_dict['seed'])
        data[metric]['model_seeds'].append(run_dict['model_seed'])

    for entry in data.values(): 
        entry['mean'] = np.array(entry['all']).mean(axis=0) 
        entry['std'] = np.array(entry['all']).std(axis=0)
    
    return data

def distributions(k): 
    with open(save_dir/f'exhaustive_metrics_{k}.pickle','rb') as f:
        metrics = pickle.load(f)

    valid_smiles = np.load(save_dir/'valid_smiles.np')
    valid_scores = np.load(save_dir/'valid_scores.np')

    targets = {
        'IGF1R': '-IGF1R',
        'EGFR': '-EGFR',
        'CYP3A4': 'CYP3A4',
    }

    for i, target in enumerate(targets.keys()): 
        fig, ax = plt.subplots(1,1)
        sns.kdeplot(data=valid_scores[:,i], ax=ax, fill=True, alpha=0.5, 
                    color=list(cluster_colors.values())[-i], linewidth=0.8,
                    bw_adjust=1.1)
        ax.set_xlabel(targets[target])
        ax.set_ylabel('Density')
        set_size(1.5, 1, ax=ax)
        fig.savefig(save_dir/f'distribution_{target}{filetype}',bbox_inches='tight', dpi=200, transparent=transp)

    return 

def top_k(run_dicts, batch_size=0.005): 
    data = extract_data(run_dicts, key='top-k-profile')

    fig, ax = plt.subplots(1,1)
    for i, (metric, entry) in enumerate(data.items()): 
        x = 100*batch_size*(np.arange(1, len(entry['mean'])+1))
        ax.plot(x, entry['mean'], label=acq_labels[metric][0], color=method_colors[metric], linewidth=0.7, marker=shapes[-i], markersize=2)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = method_colors[metric], alpha=0.3,
            )
        # ax.errorbar(xx, m, st, capsize=2, capthick=0.8, elinewidth=0.8, color='black')
    ax.grid(axis='y', linewidth=0.5)
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    legend.get_frame().set_facecolor("none")
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1]+0.05])
    ax.set_ylabel(f'Fraction of top ~{100*k}%')
    ax.set_xlabel('Percent of Library Acquired (%)')
    ax.locator_params(axis='y', nbins=5)
    ax.set_xticks(x)
    
    set_size(2,1.5, ax=ax)    
    fig.savefig(save_dir/f'top_{k}{filetype}',bbox_inches='tight', dpi=200, transparent=transp)

def hv(run_dicts, batch_size=0.005): 
    data = extract_data(run_dicts, key='hv_profile')

    fig, ax = plt.subplots(1,1)
    for i, (metric, entry) in enumerate(data.items()): 
        x = 100*batch_size*(np.arange(1, len(entry['mean'])+1))
        ax.plot(x, entry['mean'], label=acq_labels[metric][0], color=method_colors[metric], linewidth=0.7, marker=shapes[-i], markersize=2)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = method_colors[metric], alpha=0.3,
            )
        # ax.errorbar(xx, m, st, capsize=2, capthick=0.8, elinewidth=0.8, color='black')
    ax.grid(axis='y', linewidth=0.5)
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    legend.get_frame().set_facecolor("none")
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1]+0.05])
    ax.set_ylabel(f'Hypervolume Fraction')
    ax.set_xlabel('Percent of Library Acquired (%)')
    ax.locator_params(axis='y', nbins=5)
    ax.set_xticks(x)
    
    set_size(2,1.5, ax=ax)    
    fig.savefig(save_dir/f'hv{filetype}',bbox_inches='tight', dpi=200, transparent=transp)


def fraction_non_dominated(run_dicts, true_front_points, batch_size=0.005):
    for entry in run_dicts: 
        n_common_scores = [
            sum(np.equal(arr[None,:], true_front_points[:,None]).all(-1).any(0)) for arr in entry['all_scores']
        ]
        frac = np.array(n_common_scores)/len(true_front_points)
        entry['fraction_first_rank'] = frac

    data = extract_data(run_dicts, 'fraction_first_rank')

    fig, ax = plt.subplots(1,1)

    for i, (metric, entry) in enumerate(data.items()): 
        x = 100*batch_size*(np.arange(1, len(entry['mean'])+1))
        ax.plot(x, entry['mean'], label=acq_labels[metric][0], color=method_colors[metric], linewidth=0.7, marker=shapes[-i], markersize=2)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = method_colors[metric], alpha=0.3,
            )
        # ax.errorbar(xx, m, st, capsize=2, capthick=0.8, elinewidth=0.8, color='black')
    ax.grid(axis='y', linewidth=0.5)
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    legend.get_frame().set_facecolor("none")
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1]+0.05])
    ax.set_ylabel('Fraction of true front')
    ax.set_xlabel('Percent of Library Acquired (%)')
    ax.locator_params(axis='y', nbins=5)
    ax.set_xticks(x)
    
    set_size(2,1.5, ax=ax)    
    fig.savefig(save_dir/f'frac_nd_top{filetype}',bbox_inches='tight', dpi=200, transparent=transp)

def ltx(m, std): 
    return f'${m:0.2f} \pm {std:0.2f}$'
    
def end_metrics_table(run_dicts, true_front_points): 
    for entry in run_dicts: 
        n_common_scores = [
            sum(np.equal(arr[None,:], true_front_points[:,None]).all(-1).any(0)) for arr in entry['all_scores']
        ]
        frac = np.array(n_common_scores)/len(true_front_points)
        entry['fraction_first_rank'] = frac

    fracnd_data = extract_data(run_dicts, 'fraction_first_rank')
    igd_data = extract_data(run_dicts, 'overall_igd')
    topk_data = extract_data(run_dicts, 'top-k-profile')
    hv_data = extract_data(run_dicts, key='hv_profile')

    for entry in [*topk_data.values(), *hv_data.values(), *fracnd_data.values()]: 
        entry['all'] = [profile[-1] for profile in entry['all']]
        entry['mean'] = entry['mean'][-1]
        entry['std'] = entry['std'][-1]    
    
    runs = []
    for acq in ['pi', 'random']:
        runs.append({
            'Acquisition Function': acq_labels[acq][0],
            f'Top {100*k}\%': ltx(topk_data[acq]['mean'], topk_data[acq]['std']),
            'HV': ltx(hv_data[acq]['mean'], hv_data[acq]['std']),
            'IGD': ltx(igd_data[acq]['mean'], igd_data[acq]['std']),
            'Fraction of True Front': ltx(fracnd_data[acq]['mean'], fracnd_data[acq]['std']),
        })
    
    
    df = pd.DataFrame(runs)
    df = df.sort_values(by=['Acquisition Function'])
    str_df = df.to_latex(escape=False, index=False, multicolumn_format='c')
    with open(save_dir/f'end_means_table_{k}.txt','w') as f:
        f.writelines(str_df)
    return df

def iter_table_means(key='hv_profile'): 
    data = extract_data(run_dicts, key=key)

    scores = { acq_labels[metric][0]: 
        [
            ltx(entry['mean'][i], entry['std'][i]) for i in range(len(entry['mean']))
        ]
        for metric, entry in data.items()}
        
    df = pd.DataFrame(scores)
    df.index.name = 'Iteration'
    str_df = df.to_latex(escape=False, multicolumn_format='c')
    with open(save_dir/f'{key}_means_{k}.txt','w') as f:
        f.writelines(str_df)
    return df


if __name__=='__main__': 
    # top_k(run_dicts, batch_size=batch_size)
    # fraction_non_dominated(run_dicts, exhaustive_metrics['true_front_points'], batch_size=batch_size)
    end_metrics_table(run_dicts, exhaustive_metrics['true_front_points'])
    iter_table_means(key='hv_profile')
    iter_table_means(key='top-k-profile')
    # hv(run_dicts, batch_size=batch_size)
    # distributions(k=k)