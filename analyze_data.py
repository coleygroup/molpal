from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from itertools import islice
import math
from operator import itemgetter
from pathlib import Path
import pickle
import pprint
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from rdkit import Chem, DataStructs
import seaborn as sns
from tqdm import tqdm

import utils
import sar

sns.set_theme(style='white', context='paper')

class PrettyPct(float):
    def __init__(self, x):
        self.x = 100*x

    def __repr__(self):
        if self.x == 0:
            return '0.000'
        elif self.x >= 0.001:
            return f'{self.x:0.3f}'
        else:
            return f'{self.x:0.1e}'

def boltzmann(xs: Iterable[float]) -> float:
    Z = sum(math.exp(-x) for x in xs)
    return sum(x * math.exp(-x) / Z for x in xs)

def mean(xs: Iterable[float]) -> float:
    return sum(x for x in xs) / len(xs)

def var(xs: Iterable[float], x_mean: float) -> float:
    return sum((x-x_mean)**2 for x in xs) / len(xs)

def mean_and_sd(xs: Iterable[float]) -> Tuple[float, float]:
    x_mean = mean(xs)
    return x_mean, math.sqrt(var(xs, x_mean))

def recursive_conversion(nested_dict):
    if not isinstance(nested_dict, defaultdict):
        return nested_dict

    for k in nested_dict:
        sub_dict = nested_dict[k]
        nested_dict[k] = recursive_conversion(sub_dict)
    return dict(nested_dict)

def read_data(p_data, k) -> List[Tuple]:
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        # the data files are always sorted
        data = [(row[0], -float(row[1]))
                for row in islice(reader, k) if row[1]]
    
    return sorted(data, key=itemgetter(1))

def get_smis_from_data(p_data) -> Set:
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        smis = {row[0] for row in reader}
    
    return smis

def compare_results(found: List[Tuple], true: List[Tuple],
                    avg: bool = True, smis: bool = True, scores: bool = True
                    ) -> Tuple[float, float, float]:
    N = len(found)
    found_smis, found_scores = zip(*found)
    true_smis, true_scores = zip(*true)

    if avg:
        found_avg = mean(found_scores)
        true_avg = mean(true_scores)
        f_avg = found_avg / true_avg
    else:
        f_avg = None

    # if boltzmann:
    #     found_boltzmann = boltzmann(found_scores)
    #     true_boltzmann = boltzmann(true_scores)
    #     f_correct_boltzmann = found_boltzmann / true_boltzmann
    # else:
    #     f_correct_boltzmann = None

    if smis:
        found_smis = set(found_smis)
        true_smis = set(true_smis)
        correct_smis = len(found_smis & true_smis)
        f_smis = correct_smis / len(true_smis)
    else:
        f_smis = None

    if scores:
        missed_scores = Counter(true_scores)
        missed_scores.subtract(found_scores)
        n_missed_scores = sum(
            count if count > 0 else 0
            for count in missed_scores.values()
        )
        f_scores = (N - n_missed_scores) / N
    else:
        f_scores = None

    return f_avg, f_smis, f_scores

def calculate_metric_results(it_rep_results):
    avg_means_sds = [[]] * len(it_rep_results)
    smis_means_sds = [[]] * len(it_rep_results)
    scores_means_sds = [[]] * len(it_rep_results)
    for i in it_rep_results:
        rep_results = zip(*it_rep_results[i].values())
        fs_avgs, fs_smis, fs_scores = rep_results

        avg_means_sds[i] = (mean_and_sd(fs_avgs))
        smis_means_sds[i] = (mean_and_sd(fs_smis))
        scores_means_sds[i] = (mean_and_sd(fs_scores))

    avg_means, avg_sds = zip(*avg_means_sds)
    avg_means = list(map(PrettyPct, avg_means))
    avg_sds = list(map(PrettyPct, avg_sds))

    smis_means, smis_sds = zip(*smis_means_sds)
    smis_means = list(map(PrettyPct, smis_means))
    smis_sds = list(map(PrettyPct, smis_sds))

    scores_means, scores_sds = zip(*scores_means_sds)
    scores_means = list(map(PrettyPct, scores_means))
    scores_sds = list(map(PrettyPct, scores_sds))

    return ((avg_means, avg_sds),
            (smis_means, smis_sds),
            (scores_means, scores_sds))

def gather_prune_data(true_data, parent_dir, N):
    d_b_prune_data = {}

    parent_dir = Path(parent_dir)
    for batches_dir in parent_dir.iterdir():
        b = int(batches_dir.name[1:])
        d_b_prune_data[b] = {}
        for prune_dir in batches_dir.iterdir():
            prune = prune_dir.name

            data = top_N_metrics(true_data, prune_dir, N)
            # if prune=='best':
            #     pprint.pprint(data)

            d_b_prune_data[b][prune] = data
    
    # pprint.pprint(d_b_prune_data, compact=True)

    return d_b_prune_data

def gather_results(run, true_data, N) -> List[Tuple[float, float, float]]:
    data = run / 'data'

    d_it_results = {}
    for it_data in tqdm(data.iterdir(), desc='Analyzing iterations',
                        leave=False):
        try:
            it = int(it_data.stem.split('_')[-1])
        except ValueError:
            continue

        found = read_data(it_data, N)
        d_it_results[it] = compare_results(found, true_data)

    return [d_it_results[it] for it in sorted(d_it_results.keys())]

def average_it_results(it_rep_results: Sequence[Sequence[Tuple]]):
    d_metric_results = {
        'avg': [],
        'smis': [],
        'scores': []
    }
    for it_reps in it_rep_results:
        it_avgs, it_smiss, it_scoress = zip(*it_reps)
        d_metric_results['avg'].append(mean_and_sd(it_avgs))
        d_metric_results['smis'].append(mean_and_sd(it_smiss))
        d_metric_results['scores'].append(mean_and_sd(it_scoress))

    return d_metric_results

def analyze_nn_data(parent_dir, true_data, N):
    nested_dict = lambda: defaultdict(nested_dict)
    results = nested_dict()

    parent_dir = Path(parent_dir)
    for thresh in parent_dir.iterdir():
        for model in thresh.iterdir():
            for metric in model.iterdir():
                for rep in metric.iterdir():
                    rep_ = int(rep.name.split('_')[-1])
                    results[
                        thresh.name][model.name][metric.name][rep_
                    ] = gather_results(rep, true_data, N)
    results = recursive_conversion(results)

    for thresh in results:
        for model in results[thresh]:
            for metric in results[thresh][model]:
                it_rep_results = list(zip(
                    *results[thresh][model][metric].values()
                ))
                it_results = average_it_results(it_rep_results)
                results[thresh][model][metric] = it_results
                
    return results

METRICS = ['greedy', 'ucb', 'thompson', 'ei', 'pi']
METRIC_NAMES = {'greedy': 'greedy', 'ucb': 'UCB', 'thompson': 'TS',
                'ei': 'EI', 'pi': 'PI'}
METRIC_COLORS = dict(zip(METRICS, sns.color_palette('bright')))

def plot_nn_data(d_t_data, size: int, split: float, N: int,
                 model='rf', reward='scores'):
    xs = [int(size*split * i) for i in range(1, 7)]

    fig, axs = plt.subplots(1, len(d_t_data), sharex=True, sharey=True,
                            figsize=(4/1.5 * 3, 4))
    fmt = 'o-'
    ms = 5
    capsize = 2
    
    for i, t in enumerate(sorted(d_t_data.keys())):
        for metric in d_t_data[t][model]:
            ys, sds = zip(*d_t_data[t][model][metric][reward])
            ys = [y * 100 for y in ys]
            sds = [sd * 100 for sd in sds]

            _ = axs[i].errorbar(
                xs, ys, yerr=sds,
                color=METRIC_COLORS[metric], label=METRIC_NAMES[metric],
                fmt=fmt, ms=ms, mec='black', capsize=capsize
            )
        axs[i].set_title(' = '.join([t[0], t[1:]]))
        if i == 0:
            axs[i].set_ylabel(f'Percentage of Top-{N} {reward} Found')
        axs[i].set_ylim(bottom=0)

        axs[i].set_xlabel(f'Molecules explored')
        axs[i].set_xlim(left=0)
        axs[i].xaxis.set_major_locator(ticker.MaxNLocator(7))
        axs[i].xaxis.set_tick_params(rotation=30)

        axs[i].grid(True)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.081, 0.71), title='Metric')
    fig.tight_layout()

    return fig

def top_N_metrics(true_data, parent_dir, N):
    true = true_data
    parent_dir = Path(parent_dir)

    nested_dict = lambda: defaultdict(nested_dict)
    results = nested_dict()
    random = nested_dict()
    common_smis = nested_dict()

    for child in tqdm(parent_dir.iterdir(), desc='Analyzing runs', unit='run'):
        if not child.is_dir():
            continue

        fields = str(child.stem).split('_')
        _, _, model, metric, _, rep, *_ = fields
        rep = int(rep)
        # _, model, metric, _, _, repeat, *_ = fields
        
        p_data = child / 'data'
        for p_iter in tqdm(p_data.iterdir(), desc='Analyzing iterations',
                           leave=False):
            try:
                it = int(p_iter.stem.split('_')[-1])
            except ValueError:
                continue

            found = read_data(p_iter, N)
            if metric == 'random':
                random[it][rep] = compare_results(found, true)
            else:
                results[model][metric][it][rep] = compare_results(found, true)
                common_smis[model][metric][it][rep] = {smi for smi, _ in found}

    d_model_metric_it_rep_results = recursive_conversion(results)
    random = recursive_conversion(random)
    common_smis = recursive_conversion(common_smis)

    d_model_mode_metric_results = {}
    for model in d_model_metric_it_rep_results:
        d_metric_it_rep_results = d_model_metric_it_rep_results[model]

        metric_results_avg = {}
        metric_results_smis = {}
        metric_results_scores = {}

        for metric in d_metric_it_rep_results:
            d_it_rep_results = d_metric_it_rep_results[metric]

            avg, smis, scores = calculate_metric_results(d_it_rep_results)

            metric_results_avg[metric] = avg[0], avg[1]
            metric_results_smis[metric] = smis[0], smis[1]
            metric_results_scores[metric] = scores[0], scores[1]

        d_model_mode_metric_results[model] = {
            'avg': metric_results_avg,
            'smis': metric_results_smis,
            'scores': metric_results_scores
        }

    if random:
        avg, smis, scores = calculate_metric_results(random)
        random = {
            'avg': (avg[0], avg[1]),
            'smis': (smis[0], smis[1]),
            'scores': (scores[0], scores[1])
        }
    
    return d_model_mode_metric_results

def top_N_intersections(parent_dir, N):
    parent_dir = Path(parent_dir)

    nested_dict = lambda: defaultdict(nested_dict)
    common_smis = nested_dict()

    for child in tqdm(parent_dir.iterdir(), desc='Analyzing runs', unit='run'):
        if not child.is_dir():
            continue

        fields = str(child.stem).split('_')
        _, model, metric, _, rep, *_ = fields
        rep = int(rep)
        
        p_data = child / 'data'
        for p_iter in tqdm(p_data.iterdir(), desc='Analyzing iterations',
                           leave=False):
            try:
                it = int(p_iter.stem.split('_')[-1])
            except ValueError:
                continue

            found = read_data(p_iter, N)
            if metric == 'random':
                pass
            else:
                common_smis[model][metric][it][rep] = {smi for smi, _ in found}

    common_smis = recursive_conversion(common_smis)
    for model in common_smis:
        for metric in common_smis[model]:
            n_smis_by_iter = []
            for it in common_smis[model][metric]:
                smis_sets = list(common_smis[model][metric][it].values())
                n_smis_by_iter.append(len(set.intersection(*smis_sets)))
            common_smis[model][metric] = n_smis_by_iter
    d_model_metric_results = common_smis

    return d_model_metric_results

def unions(parent_dir):
    parent_dir = Path(parent_dir)

    nested_dict = lambda: defaultdict(nested_dict)
    total_smis = nested_dict()
    random = nested_dict()

    for child in tqdm(parent_dir.iterdir(), desc='Analyzing runs', unit='run'):
        if not child.is_dir():
            continue

        fields = str(child.stem).split('_')
        _, model, metric, _, rep, *_ = fields
        rep = int(rep)
        
        p_data = child / 'data'
        for p_iter in tqdm(p_data.iterdir(), desc='Analyzing iterations',
                           leave=False):
            try:
                it = int(p_iter.stem.split('_')[-1])
            except ValueError:
                continue

            if metric == 'random':
                random[it][rep] = get_smis_from_data(p_iter)
            else:
                total_smis[model][metric][it][rep] = get_smis_from_data(p_iter)

    total_smis = recursive_conversion(total_smis)
    for model in total_smis:
        for metric in total_smis[model]:
            n_smis_by_iter = []
            for it in range(6):#total_smis[model][metric]:
                smis_sets = list(total_smis[model][metric][it].values())
                n_smis_by_iter.append(len(set.union(*smis_sets)))
            total_smis[model][metric] = n_smis_by_iter
            
    random = recursive_conversion(random)
    n_random_smis_by_iter = []

    d_model_metric_results = total_smis
    return d_model_metric_results

sns.set_theme(style='white', context='paper')

MODELS = ['rf', 'nn', 'mpn']
MODEL_COLORS = dict(zip(MODELS, sns.color_palette('dark')))

# METRICS = ['greedy', 'ucb', 'thompson', 'ei', 'pi']
# METRIC_NAMES = {'greedy': 'greedy', 'ucb': 'UCB', 'thompson': 'TS',
#                 'ei': 'EI', 'pi': 'PI'}
PRUNE_METHODS = ['best', 'random', 'maxmin', 'leader']
PRUNE_COLORS = dict(zip(PRUNE_METHODS, sns.color_palette('dark')))

SPLITS = [0.4, 0.2, 0.1]

DASHES = ['dash', 'dot', 'dashdot']
MARKERS = ['circle', 'square', 'diamond']

def plot_prune_data(
        d_b_prune_data, size: int, split: float, N: int,
        model='rf', score='scores', metric='greedy'
    ):
    xs = [int(size*split * i) for i in range(1, 7)]

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                            figsize=(4/1.5 * 3, 4))
    fmt = 'o-'
    ms = 5
    capsize = 2
    
    n_batches = range(2, 5)
    for i, (b, ax) in enumerate(zip(n_batches, axs)):
        for prune in PRUNE_METHODS:
            ys, y_sds = d_b_prune_data[b][prune][model][score][metric]
            ys = [y*100 for y in ys]
            y_sds = [y*100 for y in y_sds]

            ebar = ax.errorbar(
                xs, ys, yerr=y_sds, color=PRUNE_COLORS[prune], label=prune,
                fmt=fmt, ms=ms, mec='black', capsize=capsize
            )
            # else:
                # ys, y_sds = d_b_prune_data[b][prune][model][score][metric]
                # ax.plot(xs, ys, fmt, color=METRIC_COLORS[metric],
                #         ms=ms, mec='black',
                #         alpha=0.33 if si_fig else 1.)
                # ys, y_sds = online_results[model][score][metric]
                # ebar = ax.errorbar(
                #     xs, ys, yerr=y_sds, color=METRIC_COLORS[metric],
                #     fmt=fmt, ms=ms, mec='black', capsize=capsize,
                # )
            # ebar.set_label(METRIC_NAMES[metric])
                
        # ys, y_sds = random_results[score]
        # ax.errorbar(
        #     xs, ys, yerr=y_sds, fmt=fmt, ms=ms, color='grey',
        #     mec='black', capsize=capsize, label='random'
        # )
        ax.set_title(f'B={b}')
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {score} Found')
        ax.set_ylim(bottom=0)

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.081, 0.62), title='Metric')
    fig.tight_layout()

    return fig

def parse_errors_file(errors_filepath) -> np.ndarray:
    with open(errors_filepath, 'r') as fid:
        reader = csv.reader(fid)
        next(reader)
        errors = [float(row[1]) for row in reader]

    return np.array(errors)

def gather_error_data(parent_dir):
    parent_dir = Path(parent_dir)
    p_errors = parent_dir / 'errors'

    def get_iter(p_error_iter):
        return int(p_error_iter.stem.split('_')[-1])
    errors_files = sorted(p_errors.iterdir(), key=get_iter)
    errors_by_iter = [parse_errors_file(e_file) for e_file in errors_files]

    return errors_by_iter

def plot_error_data(errors_by_iter: np.ndarray, b, prune, cumulative=True):
    fig = plt.Figure(figsize=(4/1.5 * 3, 4))
    ax = fig.add_subplot(111)

    labels = [f'iter {i}' for i in range(len(errors_by_iter))]
    if cumulative:
        errors_by_iter = [np.abs(errors) for errors in errors_by_iter]
        ax.hist(
            errors_by_iter, density=True, label=labels,
            cumulative=True, histtype='step', 
        )
        ax.set_xlabel(f'Absolute Predictive error / kcal*mol^-1')
    else:
        ax.hist(
            errors_by_iter, density=True, label=labels, #histtype='barstacked'
        )
        ax.set_xlabel(f'Predictive error / kcal*mol^-1')

    ax.set_title(f'MolPAL Predictive Errors (B={b}, {prune} pruning)')
    ax.set_ylabel('Density')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    return fig

def get_data_by_iter(parent_dir, smiles_col, score_col, N) -> List[Dict]:
    p_data = Path(parent_dir) / 'data'
    p_csvs = [p_csv for p_csv in p_data.iterdir() if 'iter' in str(p_csv)]

    data_by_iter = [
        utils.parse_csv(
            p_csv, smiles_col=smiles_col,
            score_col=score_col, k=N
        ) for p_csv in p_csvs
    ]

    return sorted(data_by_iter, key=lambda d: len(d))

def partition_data_by_iter(data_by_iter) -> List[Dict]:
    """separate all of the data by the iteration at which they were acquired"""
    new_data_by_iter = [data_by_iter[0]]
    for i in range(1, len(data_by_iter)):
        current_data = set(data_by_iter[i].items())
        previous_data = set(data_by_iter[i-1].items())
        new_data = dict(current_data ^ previous_data)
        new_data_by_iter.append(new_data)

    return new_data_by_iter

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--true-pkl')
    parser.add_argument('--parent-dir', required=True)
    parser.add_argument('--library')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('-k', type=int)
    parser.add_argument('-N', type=int)
    parser.add_argument('--split', type=float, required=True)
    parser.add_argument('--mode', required=True,
                        choices=('topk', 'errors', 'diversity', 'hists',
                                 'intersection', 'union'))
    parser.add_argument('--name', default='.')
    parser.add_argument('--format', '--fmt', default='png',
                        choices=('png', 'pdf'))
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--log', action='store_true', default=False)

    args = parser.parse_args()

    if args.true_pkl:
        true_data = pickle.load(open(args.true_pkl, 'rb'))
        size = len(true_data)
        try:
            true_data = sorted(true_data.items(), key=itemgetter(1))[:args.N]
        except AttributeError:
            true_data = sorted(true_data, key=itemgetter(1))[:args.N]

    if args.mode == 'topk':
        # d_b_prune_data = gather_prune_data(
        #     true_data, args.parent_dir, args.N
        # )
        d_t_data = analyze_nn_data(args.parent_dir, true_data, args.N)
        fig = plot_nn_data(d_t_data, size, args.split, args.N)

        results_dir = Path(f'{args.name}/{args.split}')
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{results_dir}/top{args.N}_results_plot.{args.format}')

    elif args.mode == 'errors':
        p_errors = Path(f'{args.name}') / 'errors'
        p_errors.mkdir(exist_ok=True, parents=True)
        b, prune = Path(args.parent_dir).name.split('_')[5:]

        error_data = gather_error_data(args.parent_dir)
        fig = plot_error_data(error_data, b, prune)

        fig.savefig(f'{p_errors}/{b}_{prune}_hist.{args.format}')

    elif args.mode == 'diversity':
        p_div = Path(f'{args.name}/{args.split}') / 'diversity'
        p_div.mkdir(exist_ok=True, parents=True)

        data_by_iter = get_data_by_iter(
            args.parent_dir, args.smiles_col, args.score_col, args.N
        )
        new_data_by_iter = partition_data_by_iter(data_by_iter)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        bins = np.linspace(0., 1., args.bins)
        width = (bins[1] - bins[0])

        for i, data in enumerate(new_data_by_iter):
            smis, _ = zip(*data.items())
            fps = utils.smis_to_fps(smis)
            H, logH = sar.distance_hist(fps, None, args.bins, None, args.k)
            # hist = distance_hist(data, args.bins, True)
            
            ax.bar(bins, H, align='edge', width=width,
                   label=i, alpha=0.7)

        plt.legend(title='Iteration')
        k = f'{args.k}' if args.k else 'all'

        name, model, metric, split, *_ = Path(args.parent_dir).name.split('_')
        
        base_title = f'{name} {model} {metric} {split} batch diversity'
        if args.k is None:
            comment = 'all molecules'
        elif args.k >= 1:
            comment = f'{args.k} nearest neighbors'
        else:
            comment = f'{args.k} nearest neighbor'

        ax.set_title(f'{base_title} ({comment})')
        ax.set_xlabel('Tanimoto distance')
        ax.set_ylabel('Count')

        plt.tight_layout()
        # hist_type = '2d' if args.two_d else '1d'
        # fig.savefig(f'{fig_dir}/top{args.k}_{neighbors}_{hist_type}.png')
        fig.savefig(f'{p_div}/{model}_{metric}_{k}_hist.{args.format}')

    elif args.mode == 'hists':
        results_dir = Path(f'{args.name}') / str(args.split) / 'hists'
        results_dir.mkdir(parents=True, exist_ok=True)
        path = Path(args.parent_dir)
        preds_paths = sorted(
            [p for p in path.iterdir()],
            key=lambda p: str(p).split('_')[-1]
        )

        with open(args.library, 'r') as fid:
            reader = csv.reader(fid)
            next(reader)
            smis = [row[args.smiles_col] for row in tqdm(reader)]
        fps = utils.smis_to_fps(smis)

        N = len(preds_paths)
        fig, axss = plt.subplots(N, 2, figsize=(10, 5*N))
        for i, (path, axs) in enumerate(zip(preds_paths, axss)):

            preds = np.load(path)
            try:
                preds = preds[:, 0]
            except:
                preds = preds

            range_ = [(0., 1.), (0., preds.max() - preds.min())]
            H, logH = sar.distance_hist(
                fps, preds, args.bins, range_, args.k, True
            )
            xedges = np.linspace(0., 1., args.bins)
            yedges = np.linspace(0., range_[1][1], args.bins)
            X, Y = np.meshgrid(xedges, yedges)

            pcm1 = axs[0].pcolormesh(X, Y, H, shading='auto')
            fig.colorbar(pcm1, ax=axs[0], label='Count')
            pcm2 = axs[1].pcolormesh(X, Y, logH, shading='auto')
            fig.colorbar(pcm2, ax=axs[1], label='log(Count)')
            axs[0].set_ylabel('Predicted Score Distance')

        for ax in axs:    
            ax.set_xlabel('Tanimoto Distance')

        fig.suptitle(f'{args.name} Pairwise Distance Hisograms')
        ax.set_ylabel('Predicted Score Distance')
        
        fig.savefig(f'{results_dir}/preds_hists.{args.format}')

    elif args.mode == 'clusters':
        path = Path(args.parent_dir)
        p_iters = sorted(
            [p for p in path.iterdir()],
            key=lambda p: str(p).split('_')[-1]
        )

        true_smis, _ = zip(*true_data)
        fps = utils.smis_to_fps(true_smis, args.radius, args.length)
        idxs, centroids = sar.compute_centroids(fps, args.threshold)

        for p_iter in tqdm(p_iters):
            found = read_data(p_iter, args.N)
            found_smis, _ = zip(*found)
            found_fps = utils.smis_to_fps(found_smis, args.radius, args.length)

            cids = []
            for fp in fps:
                T_cs = DataStructs.BulkTanimotoSimilarity(fp, centroids)
                T_cs = np.array(T_cs)

                i = np.argmin(T_cs)
                cids.append(idxs[i])
            

    elif args.mode == 'intersection':
        top_N_intersections(args.parent_dir, args.N)

    elif args.mode == 'union':
        unions(args.parent_dir)

    else:
        exit()