from argparse import ArgumentParser
from collections import Counter
import csv
from functools import partial
import gzip
import heapq
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

sys.path.append('..')
from molpal.acquirer import metrics

REPS = 1

sns.set_theme(style='white', context='paper')

palette = []
for cmap in ('Purples', 'Reds', 'Greens', 'Blues', 'Oranges'):
    palette.extend(sns.color_palette(cmap, REPS))

sns.set_palette(palette)

#-----------------------------------------------------------------------------#

def extract_smis(library, smiles_col=0, title_line=True) -> List:
    if Path(library).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open
    
    with open_(library) as fid:
        reader = csv.reader(fid)
        if title_line:
            next(reader)

        smis = []
        for row in reader:
            try:
                smis.append(row[smiles_col])
            except ValueError:
                continue

    return smis

def build_true_dict(true_csv, smiles_col: int = 0, score_col: int = 1,
                    title_line: bool = True,
                    maximize: bool = False) -> Dict[str, float]:
    if Path(true_csv).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open
    
    c = 1 if maximize else -1

    with open_(true_csv) as fid:
        reader = csv.reader(fid)
        if title_line:
            next(reader)

        d_smi_score = {}
        for row in reader:
            try:
                d_smi_score[row[smiles_col]] = c * float(row[score_col])
            except ValueError:
                continue

    return d_smi_score

def read_scores(scores_csv: str) -> Tuple[Dict, Dict]:
    """read the scores contained in the file located at scores_csv"""
    scores = {}
    failures = {}
    with open(scores_csv) as fid:
        reader = csv.reader(fid)
        next(reader)
        for row in reader:
            try:
                scores[row[0]] = float(row[1])
            except:
                failures[row[0]] = None
    
    return scores, failures

#-----------------------------------------------------------------------------#

def gather_experiment_predss(experiment) -> Tuple[List, List]:
    chkpts_dir = Path(experiment) / 'chkpts'

    chkpt_iter_dirs = sorted(
        chkpts_dir.iterdir(), key=lambda p: int(p.stem.split('_')[-1])
    )[1:]
    try:
        # new way
        preds_npzs = [np.load(chkpt_iter_dir / 'preds.npz')
                      for chkpt_iter_dir in chkpt_iter_dirs]
        predss, varss = zip(*[
            (preds_npz['Y_pred'], preds_npz['Y_var'])
            for preds_npz in preds_npzs
        ])

        return predss, varss
    except FileNotFoundError:
        # old way
        predss = [np.load(chkpt_iter_dir / 'preds.npy')
                  for chkpt_iter_dir in chkpt_iter_dirs]

        return predss, []

def calculate_utilties(
        metric: str, Y_preds: List[np.ndarray], Y_vars: List[np.ndarray], 
        new_points_by_epoch: List[Dict], k: int = 1,
        beta: float = 2., xi: float = 0.01
    ) -> List[np.ndarray]:
    Us = []
    explored = {}

    for Y_pred, Y_var, new_points in zip(Y_preds, Y_vars, new_points_by_epoch):
        explored.update(new_points)
        ys = list(explored.values())
        Y = np.nan_to_num(np.array(ys, dtype=float), nan=-np.inf)
        current_max = np.partition(Y, -k)[-k]

        Us.append(metrics.calc(
            metric, Y_mean=Y_pred, Y_var=Y_var,
            current_max=current_max, beta=beta, xi=xi,
        ))

    return Us

def get_new_points_by_epoch(experiment) -> List[Dict]:
    """get the set of new points acquired at each iteration in the list of 
    scores_csvs that are already sorted by iteration"""
    data_dir = Path(experiment) / 'data'

    scores_csvs = [p for p in data_dir.iterdir() if 'final' not in p.stem]
    scores_csvs = sorted(
        scores_csvs, key=lambda p: int(p.stem.split('_')[-1])
    )

    all_points = {}
    new_points_by_epoch = []
    for scores_csv in scores_csvs:
        scores, failures = read_scores(scores_csv)
        scores.update(failures)
        new_points = {smi: score for smi, score in scores.items()
                    if smi not in all_points}
        new_points_by_epoch.append(new_points)
        all_points.update(new_points)
    
    return new_points_by_epoch

def get_all_points_in_order(experiment: str, metric: str,
                            d_smi_idx: Dict) -> Tuple[int, List]:
    """Get all points acquired during a MolPAL run in the order in which they
    were acquired as well as the initialization batch size"""
    new_points_by_epoch = get_new_points_by_epoch(experiment)
    init_size = len(new_points_by_epoch[0])

    Y_preds, Y_vars = gather_experiment_predss(experiment)
    Us = calculate_utilties(metric, Y_preds, Y_vars, new_points_by_epoch)

    all_points_in_order = []
    all_points_in_order.extend(new_points_by_epoch[0].items())

    for new_points, U in zip(new_points_by_epoch[1:], Us):
        us = np.array([
            U[d_smi_idx[smi]]
            for smi in tqdm(new_points, desc='ordering', leave=False)
        ])
        new_points_in_order = [
            smi_score for _, smi_score in sorted(
                zip(us, new_points.items()), reverse=True
            )
        ]
        all_points_in_order.extend(new_points_in_order)
    
    return init_size, all_points_in_order

def reward_curve(
        all_points_in_order, true_top_k: List, reward: str = 'scores'
    ) -> np.ndarray:
    """Calculate the reward curve of a molpal run

    Parameters
    ----------
    all_points_in_order : str
        The points acquired during a MolPAL run, ordered by acquisition timing
    metric : str
        the acquisition metric used
    d_smi_idx : Dict
    true_top_k : List
        the list of the true top-k molecules as tuples of their SMILES string
        and associated score
    reward : str, default='scores'
        the type of reward to calculate

    Returns
    -------
    np.ndarray
        the reward as a function of the number of molecules sampled
    """
    k = len(true_top_k)

    if reward == 'scores':
        _, true_scores = zip(*true_top_k)
        missed_scores = Counter(true_scores)

        all_hits_in_order = np.zeros(len(all_points_in_order), dtype=bool)
        for i, (_, score) in enumerate(all_points_in_order):
            if score not in missed_scores:
                continue
            all_hits_in_order[i] = True
            missed_scores[score] -= 1
            if missed_scores[score] == 0:
                del missed_scores[score]
        reward_curve = 100 * np.cumsum(all_hits_in_order) / k

    elif reward == 'smis':
        true_top_k_smis = {smi for smi, _ in true_top_k}
        all_hits_in_order = np.array([
            smi in true_top_k_smis
            for smi, _ in all_points_in_order
        ], dtype=bool)
        reward_curve = 100 * np.cumsum(all_hits_in_order) / k

    elif reward == 'top-k-ave':
        reward_curve = np.zeros(len(all_points_in_order), dtype='f8')
        heap = []

        for i, (_, score) in enumerate(all_points_in_order[:k]):
            if score is not None:
                heapq.heappush(heap, score)
            top_k_avg = sum(heap) / k
            reward_curve[i] = top_k_avg
        reward_curve[:k] = top_k_avg

        for i, (_, score) in enumerate(all_points_in_order[k:]):
            if score is not None:
                heapq.heappushpop(heap, score)

            top_k_avg = sum(heap) / k
            reward_curve[i+k] = top_k_avg

    elif reward == 'total-ave':
        _, all_scores_in_order = zip(*all_points_in_order)
        Y = np.array(all_scores_in_order, dtype=float)
        Y = np.nan_to_num(Y)
        N = np.arange(0, len(Y)) + 1
        reward_curve = np.cumsum(Y) / N
        
    else:
        raise ValueError

    return reward_curve

#-----------------------------------------------------------------------------#

def style_axis(ax):
    ax.set_xlabel(f'Molecules sampled')
    ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0)#, top=100)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.xaxis.set_tick_params(rotation=30)
    ax.grid(True)

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e3:
        return f'{x*1e-3:0.1f}k'

    return f'{x:0.0f}'

#-----------------------------------------------------------------------------#

def plot_reward_curve(reward_curves: Iterable[np.ndarray],
                      metrics: Iterable[str], bs: int, title: str):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 4))
    fmt = '-'

    y_max = 0
    for Y, metric in zip(reward_curves, metrics):
        y_max = max(y_max, max(Y))
        X = np.arange(len(Y)) + 1
        ax.plot(X, Y, fmt, label=metric, alpha=0.9)

    n_iters = len(X) // bs
    ax.vlines([bs * (i+1) for i in range(n_iters)], 0, y_max,
              color='r', ls='dashed', lw=0.5)
    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    ax.xaxis.set_major_formatter(formatter)
    style_axis(ax)
    ax.set_ylabel(title)
    
    # ax.legend(loc='upper left', title='Metric')

    fig.tight_layout()

    return fig

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-e', '--experiments', '--expts', nargs='+',
                        help='the top-level directory generated by the MolPAL run. I.e., the directory with the "data" and "chkpts" directories')
    parser.add_argument('-l', '--library',
                        help='the library file used for the corresponding MolPAL run.')
    parser.add_argument('--true-csv',
                        help='a pickle file containing a dictionary of the true scoring data')
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('--no-title-line', action='store_true', default=False)
    parser.add_argument('--maximize', action='store_true', default=False,
                        help='whether the objective for which you are calculating performance should be maximized.')
    parser.add_argument('-m', '--metrics', nargs='+',
                        help='the respective acquisition metric used for each experiment')
    parser.add_argument('-k', type=int,
                        help='')
    parser.add_argument('-N', type=int,
                        help='the number of top scores from which to calculate performance')
    parser.add_argument('-r', '--reward',
                        choices=('scores', 'smis', 'top-k-ave', 'total-ave'),
                        help='the type of reward to calculate')
    parser.add_argument('--split', type=float, default=0.004,
                        help='the split size to plot when using model-metrics mode')
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--labels', nargs='+')
    
    args = parser.parse_args()
    args.title_line = not args.no_title_line

    smis = extract_smis(args.library, args.smiles_col, args.title_line)
    d_smi_score = build_true_dict(
        args.true_csv, args.smiles_col, args.score_col,
        args.title_line, args.maximize
    )

    true_smis_scores = sorted(d_smi_score.items(), key=lambda kv: kv[1])[::-1]
    true_top_k = true_smis_scores[:args.N]
    d_smi_idx = {smi: i for i, smi in enumerate(smis)}

    reward_curves = []
    init_sizes = []
    for experiment, metric in zip(args.experiments, args.metrics):
        init_size, all_points_in_order = get_all_points_in_order(
            experiment, metric, d_smi_idx
        )
        init_sizes.append(init_size)
        reward_curves.append(reward_curve(
            all_points_in_order, true_top_k, args.reward
        ))

    
    bs = int(args.split * len(smis))
    title = {
        'smis': f'Percentage of Top-{args.N} SMILES Found',
        'scores': f'Percentage of Top-{args.N} scores Found',
        'top-k-ave': f'Top-{args.N} average',
        'total-ave': f'Total average'
    }[args.reward]

    fig = plot_reward_curve(
        reward_curves, args.metrics, bs, title
    ).savefig(args.names[0])

    exit()