from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from itertools import islice
from operator import itemgetter
from pathlib import Path
import pickle
import pprint
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

METRICS = ['greedy', 'ucb', 'ts', 'ei', 'pi']
METRIC_NAMES = {'greedy': 'greedy', 'ucb': 'UCB', 'ts': 'TS',
                'ei': 'EI', 'pi': 'PI'}
METRIC_COLORS = dict(zip(METRICS, sns.color_palette('bright')))

MODELS = ['rf', 'nn', 'mpn']
MODEL_COLORS = dict(zip(MODELS, sns.color_palette('dark')))

SPLITS = [0.004, 0.002, 0.001]

DASHES = ['dash', 'dot', 'dashdot']
MARKERS = ['circle', 'square', 'diamond']

BETAS = [0.1, 0.5, 1, 2, 5, 10]
BETA_COLORS = dict(zip(BETAS, sns.color_palette('Oranges')))

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
        data = [(row[0], c * float(row[1]))
                for row in islice(reader, k) if row[1]]
    
    return data

def calculate_rewards(found: List[Tuple], true: List[Tuple],
                      avg: bool = True, smis: bool = True, scores: bool = True
                      ) -> Tuple[float, float, float]:
    N = len(found)
    found_smis, found_scores = zip(*found)
    true_smis, true_scores = zip(*true)

    if avg:
        found_avg = np.mean(found_scores)
        true_avg = np.mean(true_scores)
        f_avg = found_avg / true_avg
    else:
        f_avg = None

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

    return [d_it_results[it] for it in sorted(d_it_results.keys())]

def gather_beta_results(beta, true_data, N, maximize: bool = False):
    resultss = [
        gather_run_results(rep, true_data, N, maximize)
        for rep in tqdm(beta.iterdir(), 'Reps', None, False)
    ]
    resultss = [results for results in resultss if len(results) == 6]
    resultss = np.array(resultss)

    means = np.mean(resultss, axis=0)
    sds = np.sqrt(np.var(resultss, axis=0))

    try:
        return {
            'avg': list(zip(means[:, 0], sds[:, 0])),
            'smis': list(zip(means[:, 1], sds[:, 1])),
            'scores': list(zip(means[:, 2], sds[:, 2]))
        }
    except:
        return {
            'avg': list(zip(np.zeros(6), np.zeros(6))),
            'smis': list(zip(np.zeros(6), np.zeros(6))),
            'scores': list(zip(np.zeros(6), np.zeros(6)))
        }

def gather_all_beta_rewards(
        parent_dir, true_data, N: int,
        overwrite: bool = False, maximize: bool = False
    ) -> Dict:
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
                for beta in tqdm((model/'ucb').iterdir(), 'Betas', None, False):
                    results[training.name][
                        float(split.name)][
                        model.name][
                        float(beta.name)
                    ] = gather_beta_results(beta, true_data, N, maximize)
    results = recursive_conversion(results)

    pickle.dump(results, open(cached_rewards, 'wb'))

    return results

################################################################################
#------------------------------------------------------------------------------#
################################################################################

def style_axis(ax):
    ax.set_xlabel(f'Molecules explored')
    ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0, top=100)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.xaxis.set_tick_params(rotation=30)
    ax.grid(True)

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e3:
        return f'{x*1e-3:0.0f}k'

    return f'{x:0.0f}'

def plot_betas(results, size: int, N: int,
               split: float = 0.001, model: str = 'mpn', reward='scores'):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4/1.5, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2

    for beta in BETAS[::-1]:
        try:
            ys, y_sds = zip(*results['retrain'][split][model][beta][reward])
        except:
            continue
        
        ys = [y*100 for y in ys]
        y_sds = [y*100 for y in y_sds]
        xs = [int(size*split * (i+1)) for i in range(len(ys))]

        ax.errorbar(
            xs, ys, yerr=y_sds, color=BETA_COLORS[beta],
            fmt=fmt, ms=ms, mec='black', capsize=capsize, label=beta
        )
        ax.annotate(f'{ys[-1]:0.1f}', (xs[-1],ys[-1]))

        print(f'beta: {beta} \t| final: {ys[-1]:0.2f}% (± {y_sds[-1]:0.2f})')
    # ys, y_sds = zip(*results['retrain'][split][model]['greedy'][reward])

    # ys = [y*100 for y in ys]
    # y_sds = [y*100 for y in y_sds]
    # xs = [int(size*split * (i+1)) for i in range(len(ys))]

    # ax.errorbar(
    #     xs, ys, yerr=y_sds, color='blue',
    #     fmt=fmt, ms=ms, mec='black', capsize=capsize, label='greedy'
    # )

    ax.annotate(f'{ys[-1]:0.1f}', (xs[-1],ys[-1]))
    ax.set_title(f'{split*100:0.1f}%')
    ax.set_ylabel(f'Percentage of Top-{N} {reward} Found')
    ax.legend(loc='upper left', title='Beta')

    style_axis(ax)
    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()

    return fig

################################################################################
#------------------------------------------------------------------------------#
################################################################################

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--true-pkl',
                        help='a pickle file containing a dictionary of the true scoring data')
    parser.add_argument('--size', type=int,
                        help='the size of the full library which was explored. You only need to specify this if you are using a truncated pickle file. I.e., your pickle file contains only the top 1000 scores because you only intend to calculate results of the top-k, where k <= 1000')
    parser.add_argument('--parent-dir',
                        help='the parent directory containing all of the results. NOTE: the directory must be organized in the folowing manner: <root>/<online,retrain>/<split_size>/<model>/<metric>/<repeat>/<run>. See the README for a visual description.')
    parser.add_argument('--parent-dir-sb',
                        help='the parent directory of the single batch data')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('-N', type=int,
                        help='the number of top scores from which to calculate perforamnce')
    parser.add_argument('--split', type=float,
                        help='the split size for which to generate the set of plots')
    # parser.add_argument('--mode', required=True,
    #                     choices=('model-metrics', 'split-models', 
    #                              'split-metrics', 'si', 'single-batch', 'convergence', 'csv', 'errors', 
    #                              'diversity', 'intersection'),
    #                     help='what figure to generate. For "x-y" modes, this corresponds to the figure structure, where there will be a separate panel for each "x" and in each panel there will be traces corresponding to each independent "y". E.g., "model-metrics" makes a figure with three sepearate panels, one for each model and inside each panel a trace for each metric. "si" will make the trajectory plots present in the SI.')
    # parser.add_argument('--name', default='.')
    # parser.add_argument('--format', '--fmt', default='png',
    #                     choices=('png', 'pdf'))
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='whether to overwrite the hidden cache file. This is useful if there is new data in PARENT_DIR.')
    parser.add_argument('--maximize', action='store_true', default=False,
                        help='whether the objective for which you are calculating performance should be maximized.')

    args = parser.parse_args()

    true_data = pickle.load(open(args.true_pkl, 'rb'))
    size = args.size or len(true_data)
    try:
        true_data = sorted(true_data.items(), key=itemgetter(1))
    except AttributeError:
        true_data = sorted(true_data, key=itemgetter(1))

    if args.maximize:
        true_data = true_data[::-1]
    true_data = true_data[:args.N]

    results = gather_all_beta_rewards(
        args.parent_dir, true_data, args.N, args.overwrite, args.maximize
    )
    # pprint.pprint(results, compact=True)

    fig = plot_betas(results, size, args.N)
    
    name = input('Figure name: ')
    fig.savefig(f'figures/updates/{name}.pdf')