from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from itertools import islice
import math
from operator import itemgetter
from pathlib import Path
import pickle
import pprint
from timeit import default_timer
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

METRICS = ['greedy', 'ucb', 'ts', 'ei', 'pi']
METRIC_NAMES = {'greedy': 'greedy', 'ucb': 'UCB', 'ts': 'TS',
                'ei': 'EI', 'pi': 'PI'}
METRIC_COLORS = dict(zip(METRICS, sns.color_palette('bright')))

MODELS = ['rf', 'nn', 'mpn']
MODEL_COLORS = dict(zip(MODELS, sns.color_palette('dark')))

PRUNE_METHODS = ['best', 'random', 'maxmin', 'leader']
PRUNE_COLORS = dict(zip(PRUNE_METHODS, sns.color_palette('dark')))

SPLITS = [0.004, 0.002, 0.001]

DASHES = ['dash', 'dot', 'dashdot']
MARKERS = ['circle', 'square', 'diamond']

class Timer:
    def __enter__(self):
        self.start = default_timer()
    def __exit__(self, type, value, traceback):
        self.stop = default_timer()
        print(f'{self.stop - self.start:0.4f}s')

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

def read_data(p_data, k, maximize: bool = False) -> List[Tuple]:
    c = 1 if maximize else -1
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        # the data files are always sorted
        data = [(row[0], c * float(row[1]))
                for row in islice(reader, k) if row[1]]
    
    return data

def get_smis_from_data(p_data) -> Set:
    with open(p_data) as fid:
        reader = csv.reader(fid); next(reader)
        smis = {row[0] for row in reader}
    
    return smis

def calculate_rewards(found: List[Tuple], true: List[Tuple],
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
        # true_scores = [round(x, 2) for x in true_scores]
        # found_scores = [round(x, 2) for x in found_scores]
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

# def calculate_metric_results(it_rep_results):
#     avg_means_sds = [[]] * len(it_rep_results)
#     smis_means_sds = [[]] * len(it_rep_results)
#     scores_means_sds = [[]] * len(it_rep_results)
#     for i in it_rep_results:
#         rep_results = zip(*it_rep_results[i].values())
#         fs_avgs, fs_smis, fs_scores = rep_results

#         avg_means_sds[i] = (mean_and_sd(fs_avgs))
#         smis_means_sds[i] = (mean_and_sd(fs_smis))
#         scores_means_sds[i] = (mean_and_sd(fs_scores))

#     avg_means, avg_sds = zip(*avg_means_sds)
#     avg_means = list(map(PrettyPct, avg_means))
#     avg_sds = list(map(PrettyPct, avg_sds))

#     smis_means, smis_sds = zip(*smis_means_sds)
#     smis_means = list(map(PrettyPct, smis_means))
#     smis_sds = list(map(PrettyPct, smis_sds))

#     scores_means, scores_sds = zip(*scores_means_sds)
#     scores_means = list(map(PrettyPct, scores_means))
#     scores_sds = list(map(PrettyPct, scores_sds))

#     return ((avg_means, avg_sds),
#             (smis_means, smis_sds),
#             (scores_means, scores_sds))

def gather_run_results(
        run, true_data, N, maximize: bool = False
    ) -> List[Tuple[float, float, float]]:
    data = run / 'data'

    d_it_results = {}
    for it_data in tqdm(data.iterdir(), 'Iters', None, False):
        try:
            it = int(it_data.stem.split('_')[-1])
        except ValueError:
            continue

        found = read_data(it_data, N, maximize)
        d_it_results[it] = calculate_rewards(found, true_data)

    return [(d_it_results[it]) for it in sorted(d_it_results.keys())]

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

def gather_all_rewards(parent_dir, true_data, N: int,
                       overwrite: bool = False, maximize: bool = False):
    nested_dict = lambda: defaultdict(nested_dict)
    results = nested_dict()

    parent_dir = Path(parent_dir)
    cached_rewards = parent_dir / f'.all_rewards_{N}.pkl'
    if cached_rewards.exists() and not overwrite:
        return pickle.load(open(cached_rewards, 'rb'))

    for training in tqdm(parent_dir.iterdir(), 'Training', None, False):
        if not training.is_dir():
            continue

        for split in tqdm(training.iterdir(), 'Splits', None, False):
            for model in tqdm(split.iterdir(), 'Models', None, False):
                if model.name == 'random':
                    for rep in model.iterdir():
                        rep_ = int(rep.name.split('_')[-1])
                        results[
                            training.name][
                            float(split.name)][
                            model.name][
                            rep_
                        ] = gather_run_results(rep, true_data, N)
                    continue

                for metric in tqdm(model.iterdir(), 'Metrics', None, False):
                    if metric.name == 'thompson':
                        metric_ = 'ts'
                    else:
                        metric_ = metric.name
                    for rep in tqdm(metric.iterdir(), 'Reps', None, False):
                        rep_ = int(rep.name.split('_')[-1])
                        results[
                            training.name][
                            float(split.name)][
                            model.name][
                            metric_ ][
                            rep_
                        ] = gather_run_results(rep, true_data, N, maximize)
    results = recursive_conversion(results)

    for training in results:
        for split in results[training]:
            for model in results[training][split]:
                if model == 'random':
                    rep_it_results = list(
                        results[training][split][model].values()
                    )
                    it_rep_results = list(zip(*rep_it_results))
                    it_results = average_it_results(it_rep_results)
                    results[training][split][model] = it_results
                    continue

                for metric in results[training][split][model]:
                    rep_it_results = list(
                        results[training][split][model][metric].values()
                    )
                    it_rep_results = list(zip(*rep_it_results))
                    # print(it_rep_results)
                    it_results = average_it_results(it_rep_results)
                    results[training][split][model][metric] = it_results

    pickle.dump(results, open(cached_rewards, 'wb'))

    return results

def plot_model_metrics_rewards(
        results, size: int, N: int,
        split: float = 0.010, reward='scores', si_fig: bool = False
    ):
    xs = [int(size*split * i) for i in range(1, 7)]

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                            figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2
        
    for i, (model, ax) in enumerate(zip(MODELS, axs)):
        for metric in results['retrain'][split][model]:
            if not si_fig:
                ys, y_sds = zip(
                    *results['retrain'][split][model][metric][reward]
                )
                ys = [y*100 for y in ys]
                y_sds = [y*100 for y in y_sds]
                ax.errorbar(
                    xs, ys, yerr=y_sds, color=METRIC_COLORS[metric], 
                    label=metric, fmt=fmt, ms=ms, mec='black', capsize=capsize
                )
            else:
                ys, y_sds = zip(
                    *results['retrain'][split][model][metric][reward]
                )
                ys = [y*100 for y in ys]
                y_sds = [y*100 for y in y_sds]
                ax.plot(
                    xs, ys, fmt, color=METRIC_COLORS[metric], 
                    ms=ms, mec='black', alpha=0.33,
                )
                ys, y_sds = zip(
                    *results['online'][split][model][metric][reward]
                )
                ys = [y*100 for y in ys]
                y_sds = [y*100 for y in y_sds]
                ax.errorbar(
                    xs, ys, yerr=y_sds, color=METRIC_COLORS[metric],
                    fmt=fmt, ms=ms, mec='black', capsize=capsize, label=metric
                )
        
        add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize)

        ax.set_title(model.upper())
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {reward} Found')
            ax.legend(loc=(0.06, 0.53), title='Metric')
        ax.set_ylim(bottom=0)

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.set_ylim(top=100)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    fig.tight_layout()

    return fig

def plot_split_models_rewards(
        results, size: int, N: int, metric: str = 'greedy', reward='scores'
    ):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2
        
    for i, (split, ax) in enumerate(zip(SPLITS, axs)):
        xs = [int(size*split * i) for i in range(1, 7)]

        for model in MODELS:
            if model not in results['retrain'][split]:
                continue

            ys, y_sds = zip(*results['retrain'][split][model][metric][reward])
            ys = [y*100 for y in ys]
            y_sds = [y*100 for y in y_sds]

            if len(xs) != len(ys):
                continue

            ax.errorbar(
                xs, ys, yerr=y_sds, color=MODEL_COLORS[model],
                label=model.upper(), fmt=fmt, ms=ms, mec='black', 
                capsize=capsize
            )
        
        add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize)

        ax.set_title(f'{split*100:0.1f}%')
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {reward} Found')
            ax.legend(loc=(0.05, 0.65), title='Model')
        ax.set_ylim(bottom=0)

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.set_ylim(top=100)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    fig.tight_layout()
    return fig

def plot_split_metrics_rewards(
        results, size: int, N: int,
        model: str = 'rf', reward='scores', si_fig: bool = False
    ):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2

    for i, (split, ax) in enumerate(zip(SPLITS, axs)):
        xs = [int(size*split * i) for i in range(1, 7)]

        for metric in results['retrain'][split][model]:
            ys, y_sds = zip(*results['retrain'][split][model][metric][reward])
            ys = [y*100 for y in ys]
            y_sds = [y*100 for y in y_sds]

            if len(xs) != len(ys):
                continue

            ax.errorbar(
                xs, ys, yerr=y_sds, color=METRIC_COLORS[metric], label=metric,
                fmt=fmt, ms=ms, mec='black', capsize=capsize
            )
        
        add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize)

        ax.set_title(f'{split*100:0.1f}%')
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{N} {reward} Found')
            ax.legend(loc=(0.05, 0.7), title='Metric')
        ax.set_ylim(bottom=0)

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.set_ylim(top=100)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    fig.tight_layout()
    return fig

def add_random_trace(ax, results, split, reward, xs, fmt, ms, capsize):
    try:
        try:
            ys, y_sds = zip(*results['retrain'][split]['random'][reward])
        except KeyError:
            ys, y_sds = zip(*results['online'][split]['random'][reward])
    except KeyError:
        return

    ys = [y*100 for y in ys]
    y_sds = [y*100 for y in y_sds]
    ax.errorbar(
        xs, ys, yerr=y_sds, fmt=fmt, ms=ms, color='grey',
        mec='black', capsize=capsize, label='random'
    )

def write_reward_csv(rewards, split):
    results_df = []
    for training in rewards:
        for model in rewards[training][split]:
            if model == 'random':
                scores = rewards[training][split][model]['scores'][-1]
                smis = rewards[training][split][model]['smis'][-1]
                avg = rewards[training][split][model]['avg'][-1]

                random_results = {
                    'Training': training,
                    'Model': 'random',
                    'Metric': 'random',
                    'Scores ($\pm$ s.d.)': f'{100*scores[0]:0.2f} ({100*scores[1]:0.2f})',
                    'SMILES ($\pm$ s.d.)': f'{100*smis[0]:0.2f} ({100*smis[1]:0.2f})',
                    'Average ($\pm$ s.d.)': f'{100*avg[0]:0.2f} ({100*avg[1]:0.2f})'
                }
                continue

            for metric in rewards[training][split][model]:
                if metric == 'greedy':
                    metric_ = metric.capitalize()
                elif metric == 'thompson':
                    metric_ = 'TS'
                else:
                    metric_ = metric.upper()

                scores = rewards[training][split][model][metric]['scores'][-1]
                smis = rewards[training][split][model][metric]['smis'][-1]
                avg = rewards[training][split][model][metric]['avg'][-1]

                results_df.append({
                    'Training': training,
                    'Model': model.upper(),
                    'Metric': metric_,
                    'Scores ($\pm$ s.d.)': f'{100*scores[0]:0.1f} ({100*scores[1]:0.1f})',
                    'SMILES ($\pm$ s.d.)': f'{100*smis[0]:0.1f} ({100*smis[1]:0.1f})',
                    'Average ($\pm$ s.d.)': f'{100*avg[0]:0.2f} ({100*avg[1]:0.2f})'
                })

    try:
        results_df.append(random_results)
    except UnboundLocalError:
        pass

    df = pd.DataFrame(results_df).set_index(['Training', 'Model', 'Metric'])
    return df

# def top_N_intersections(parent_dir, N):
#     parent_dir = Path(parent_dir)

#     nested_dict = lambda: defaultdict(nested_dict)
#     common_smis = nested_dict()

#     for child in tqdm(parent_dir.iterdir(), desc='Analyzing runs', unit='run'):
#         if not child.is_dir():
#             continue

#         fields = str(child.stem).split('_')
#         _, model, metric, _, rep, *_ = fields
#         rep = int(rep)
        
#         p_data = child / 'data'
#         for p_iter in tqdm(p_data.iterdir(), desc='Analyzing iterations',
#                            leave=False):
#             try:
#                 it = int(p_iter.stem.split('_')[-1])
#             except ValueError:
#                 continue

#             found = read_data(p_iter, N)
#             if metric == 'random':
#                 pass
#             else:
#                 common_smis[model][metric][it][rep] = {smi for smi, _ in found}

#     common_smis = recursive_conversion(common_smis)
#     for model in common_smis:
#         for metric in common_smis[model]:
#             n_smis_by_iter = []
#             for it in common_smis[model][metric]:
#                 smis_sets = list(common_smis[model][metric][it].values())
#                 n_smis_by_iter.append(len(set.intersection(*smis_sets)))
#             common_smis[model][metric] = n_smis_by_iter
#     d_model_metric_results = common_smis

#     return d_model_metric_results

def gather_run_smis(run) -> List[Set[str]]:
    data = run / 'data'

    d_it_smis = {}
    for it_data in tqdm(data.iterdir(), desc='Analyzing iterations',
                        leave=False, disable=True):
        try:
            it = int(it_data.stem.split('_')[-1])
        except ValueError:
            continue

        d_it_smis[it] = get_smis_from_data(it_data)

    return [d_it_smis[it] for it in sorted(d_it_smis.keys())]

def gather_smis_unions(parent_dir):
    nested_dict = lambda: defaultdict(nested_dict)
    total_smis = nested_dict()

    parent_dir = Path(parent_dir)
    cached_smis_unions = parent_dir / '.smis_unions.pkl'
    if cached_smis_unions.exists():
        return pickle.load(open(cached_smis_unions, 'rb'))

    for training in parent_dir.iterdir():
        for split in training.iterdir():
            for model in tqdm(split.iterdir(), desc='Models', leave=False):
                if model.name == 'random':
                    for rep in model.iterdir():
                        rep_ = int(rep.name.split('_')[-1])
                        total_smis[
                            training.name][
                            float(split.name)][
                            model.name][
                            rep_
                        ] = gather_run_smis(rep)
                    continue

                for metric in tqdm(model.iterdir(), 'Metrics', None, False):
                    if metric.name == 'thompson':
                        metric_ = 'ts'
                    else:
                        metric_ = metric.name
                    for rep in metric.iterdir():
                        rep_ = int(rep.name.split('_')[-1])
                        total_smis[
                            training.name][
                            float(split.name)][
                            model.name][
                            metric_ ][
                            rep_
                        ] = gather_run_smis(rep)
    total_smis = recursive_conversion(total_smis)

    for training in total_smis:
        for split in total_smis[training]:
            for model in total_smis[training][split]:
                if model == 'random':
                    rep_it_results = list(
                        total_smis[training][split][model].values()
                    )
                    it_rep_results = list(zip(*rep_it_results))
                    it_results = [
                        len(set.union(*rep_results))
                        for rep_results in it_rep_results
                    ]
                    total_smis[training][split][model] = it_results
                    continue

                for metric in total_smis[training][split][model]:
                    rep_it_results = list(
                        total_smis[training][split][model][metric].values()
                    )
                    it_rep_results = list(zip(*rep_it_results))
                    it_results = [
                        len(set.union(*rep_results))
                        for rep_results in it_rep_results
                    ]
                    total_smis[training][split][model][metric] = it_results

    pickle.dump(total_smis, open(cached_smis_unions, 'wb'))

    return total_smis

def plot_unions(results, size, metric: str = 'greedy'):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5

    for i, (split, ax) in enumerate(zip(SPLITS, axs)):
        xs = [int(size*split * i) for i in range(1, 7)]

        add_bounds(ax, xs)
        add_random(ax, xs, results, split, fmt, ms)

        for model in MODELS:
            if model not in results['retrain'][split]:
                continue
            ys = results['retrain'][split][model][metric]
            ax.plot(
                xs, ys, fmt, color=MODEL_COLORS[model],
                label=model.upper(), ms=ms, mec='black'
            )

        ax.set_title(f'{split*100:0.1f}%')
        if i == 0:
            ax.set_ylabel(f'Total Number of Unique SMILES')
            ax.legend(loc=(0.077, 0.62), title='Model')

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    fig.tight_layout()
    return fig

def plot_unions_10k50k(results_10k, results_50k, metric: str = 'greedy'):
    fig, axs = plt.subplots(1, 2, figsize=(4/1.5 * 2, 4))

    fmt = 'o-'
    ms = 5
    split = 0.010

    resultss = [results_10k, results_50k]
    sizes = [10560, 50240]
    titles = ['10k', '50k']

    for i, results in enumerate(resultss):
        ax = axs[i]
        size = sizes[i]

        xs = [int(size*split * i) for i in range(1, 7)]
        add_bounds(ax, xs)
        add_random(ax, xs, results, split, fmt, ms)

        for model in MODELS:
            if model not in results['retrain'][split]:
                continue
            ys = results['retrain'][split][model][metric]
            ax.plot(
                xs, ys, fmt, color=MODEL_COLORS[model],
                label=model.upper(), ms=ms, mec='black'
            )
        
        ax.set_title(titles[i])
        if i == 0:
            ax.set_ylabel(f'Total Number of Unique SMILES')
            ax.legend(loc=(0.077, 0.62), title='Model')

        ax.set_xlabel(f'Molecules explored')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
        ax.xaxis.set_tick_params(rotation=30)

        ax.grid(True)
    
    fig.tight_layout()
    return fig

def add_random(ax, xs, results, split, fmt, ms):
    try:
        ys_random = results['retrain'][split]['random']
    except KeyError:
        ys_random = results['online'][split]['random']
    ax.plot(
        xs, ys_random, fmt, color='grey',
        label='random', ms=ms, mec='black'
    )

def add_bounds(ax, xs):
    ys_upper = [5 * x for x in xs]
    ys_lower = [x + 4*xs[0] for x in xs]
    ax.plot( xs, ys_upper, '-', color='black')
    ax.plot(xs, ys_lower, '-', color='black')

# def parse_errors_file(errors_filepath) -> np.ndarray:
#     with open(errors_filepath, 'r') as fid:
#         reader = csv.reader(fid)
#         next(reader)
#         errors = [float(row[1]) for row in reader]

#     return np.array(errors)

# def gather_error_data(parent_dir):
#     parent_dir = Path(parent_dir)
#     p_errors = parent_dir / 'errors'

#     def get_iter(p_error_iter):
#         return int(p_error_iter.stem.split('_')[-1])
#     errors_files = sorted(p_errors.iterdir(), key=get_iter)
#     errors_by_iter = [parse_errors_file(e_file) for e_file in errors_files]

#     return errors_by_iter

# def plot_error_data(errors_by_iter: np.ndarray, b, prune, cumulative=True):
#     fig = plt.Figure(figsize=(4/1.5 * 3, 4))
#     ax = fig.add_subplot(111)

#     labels = [f'iter {i}' for i in range(len(errors_by_iter))]
#     if cumulative:
#         errors_by_iter = [np.abs(errors) for errors in errors_by_iter]
#         ax.hist(
#             errors_by_iter, density=True, label=labels,
#             cumulative=True, histtype='step', 
#         )
#         ax.set_xlabel(f'Absolute Predictive error / kcal*mol^-1')
#     else:
#         ax.hist(
#             errors_by_iter, density=True, label=labels, #histtype='barstacked'
#         )
#         ax.set_xlabel(f'Predictive error / kcal*mol^-1')

#     ax.set_title(f'MolPAL Predictive Errors (B={b}, {prune} pruning)')
#     ax.set_ylabel('Density')
#     ax.grid(True)
#     ax.legend()

#     plt.tight_layout()

#     return fig

# def get_data_by_iter(parent_dir, smiles_col, score_col, N) -> List[Dict]:
#     p_data = Path(parent_dir) / 'data'
#     p_csvs = [p_csv for p_csv in p_data.iterdir() if 'iter' in str(p_csv)]

#     data_by_iter = [
#         utils.parse_csv(
#             p_csv, smiles_col=smiles_col,
#             score_col=score_col, k=N
#         ) for p_csv in p_csvs
#     ]

#     return sorted(data_by_iter, key=lambda d: len(d))

# def partition_data_by_iter(data_by_iter) -> List[Dict]:
#     """separate all of the data by the iteration at which they were acquired"""
#     new_data_by_iter = [data_by_iter[0]]
#     for i in range(1, len(data_by_iter)):
#         current_data = set(data_by_iter[i].items())
#         previous_data = set(data_by_iter[i-1].items())
#         new_data = dict(current_data ^ previous_data)
#         new_data_by_iter.append(new_data)

#     return new_data_by_iter

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--true-pkl',
                        help='a pickle file containing a dictionary of the true scoring data')
    parser.add_argument('--size', type=int,
                        help='the size of the full library which was explored. You only need to specify this if you are using a truncated pickle file. I.e., your pickle file contains only the top 1000 scores because you only intend to calculate results of the top-k, where k <= 1000')
    parser.add_argument('--parent-dir',
                        help='the parent directory containing all of the results. NOTE: the directory must be organized in the folowing manner: <root>/<online,retrain>/<split_size>/<model>/<metric>/<repeat>/<run>. See the README for a visual description.')
    parser.add_argument('--parent-dir-10k',
                        help='the parent directory of the 10k data to make the union plot of the 10k and 50k data')
    parser.add_argument('--parent-dir-50k',
                        help='the parent directory of the 50k data to make the union plot of the 10k and 50k data')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('-k', type=int, help='the number of neighbors')
    parser.add_argument('-N', type=int,
                        help='the number of top scores from which to calculate perforamnce')
    parser.add_argument('--split', type=float,
                        help='the split size for which to generate the set of plots')
    parser.add_argument('--mode', required=True,
                        choices=('model-metrics', 'split-models', 
                                 'split-metrics', 'si', 'csv', 'errors', 
                                 'diversity', 'intersection',
                                 'union', 'union-10k50k'),
                        help='what figure to generate. For "x-y" modes, this corresponds to the figure structure, where there will be a separate panel for each "x" and in each panel there will be traces corresponding to each independent "y". E.g., "model-metrics" makes a figure with three sepearate panels, one for each model and inside each panel a trace for each metric. "si" will make the trajectory plots present in the SI.')
    # parser.add_argument('--name', default='.')
    # parser.add_argument('--format', '--fmt', default='png',
    #                     choices=('png', 'pdf'))
    # parser.add_argument('--bins', type=int, default=50)
    # parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='whether to overwrite the hidden cache file. This is useful if there is new data in PARENT_DIR.')
    parser.add_argument('--maximize', action='store_true', default=False,
                        help='whether the objective for which you are calculating performance should be maximized.')

    args = parser.parse_args()

    if args.true_pkl:
        true_data = pickle.load(open(args.true_pkl, 'rb'))
        size = args.size or len(true_data)
        try:
            true_data = sorted(true_data.items(), key=itemgetter(1))
        except AttributeError:
            true_data = sorted(true_data, key=itemgetter(1))

        if args.maximize:
            true_data = true_data[::-1]
        true_data = true_data[:args.N]

    if args.mode in ('model-metrics', 'split-models',
                     'split-metrics', 'si', 'csv'):
        results = gather_all_rewards(
            args.parent_dir, true_data, args.N, args.overwrite, args.maximize
        )

    if args.mode == 'model-metrics':
        fig = plot_model_metrics_rewards(
            results, size, args.N, args.split, 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'split-models':
        fig = plot_split_models_rewards(
            results, size, args.N, 'greedy', 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')
    
    elif args.mode == 'split-metrics':
        fig = plot_split_metrics_rewards(
            results, size, args.N, 'mpn', 'scores'
        )

        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'si':
        fig = plot_model_metrics_rewards(
            results, size, args.N, args.split, 'scores', True
        )
        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'csv':
        df = write_reward_csv(results, args.split)

        name = input('CSV name: ')
        df.to_csv(f'paper/csv/{name}.csv')

    # elif args.mode == 'errors':
    #     p_errors = Path(f'{args.name}') / 'errors'
    #     p_errors.mkdir(exist_ok=True, parents=True)
    #     b, prune = Path(args.parent_dir).name.split('_')[5:]

    #     error_data = gather_error_data(args.parent_dir)
    #     fig = plot_error_data(error_data, b, prune)

    #     fig.savefig(f'{p_errors}/{b}_{prune}_hist.{args.format}')

    # elif args.mode == 'diversity':
    #     p_div = Path(f'{args.name}/{args.split}') / 'diversity'
    #     p_div.mkdir(exist_ok=True, parents=True)

    #     data_by_iter = get_data_by_iter(
    #         args.parent_dir, args.smiles_col, args.score_col, args.N
    #     )
    #     new_data_by_iter = partition_data_by_iter(data_by_iter)

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)

    #     bins = np.linspace(0., 1., args.bins)
    #     width = (bins[1] - bins[0])

    #     for i, data in enumerate(new_data_by_iter):
    #         smis, _ = zip(*data.items())
    #         fps = utils.smis_to_fps(smis)
    #         hist = sar.distance_hist(fps, args.bins, args.k, args.log)
    #         # hist = distance_hist(data, args.bins, True)
            
    #         ax.bar(bins, hist, align='edge', width=width,
    #                label=i, alpha=0.7)

    #     plt.legend(title='Iteration')
    #     k = f'{args.k}' if args.k else 'all'

    #     name, model, metric, split, *_ = Path(args.parent_dir).name.split('_')
        
    #     base_title = f'{name} {model} {metric} {split} batch diversity'
    #     if args.k is None:
    #         comment = 'all molecules'
    #     elif args.k >= 1:
    #         comment = f'{args.k} nearest neighbors'
    #     else:
    #         comment = f'{args.k} nearest neighbor'

    #     ax.set_title(f'{base_title} ({comment})')
    #     ax.set_xlabel('Tanimoto distance')
    #     ax.set_ylabel('Count')

    #     plt.tight_layout()
    #     # hist_type = '2d' if args.two_d else '1d'
    #     # fig.savefig(f'{fig_dir}/top{args.k}_{neighbors}_{hist_type}.png')
    #     fig.savefig(f'{p_div}/{model}_{metric}_{k}_hist.{args.format}')

    # elif args.mode == 'intersection':
    #     top_N_intersections(args.parent_dir, args.N)

    elif args.mode == 'union':
        results = gather_smis_unions(args.parent_dir)
        fig = plot_unions(results, size, 'greedy')
        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    elif args.mode == 'union-10k50k':
        results_10k = gather_smis_unions(args.parent_dir_10k)
        results_50k = gather_smis_unions(args.parent_dir_50k)
        fig = plot_unions_10k50k(results_10k, results_50k, 'greedy')
        name = input('Figure name: ')
        fig.savefig(f'paper/figures/{name}.pdf')

    else:
        exit()