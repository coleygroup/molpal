import sys
sys.path.append('../molpal')

from argparse import ArgumentParser
from collections import Counter
import heapq
from itertools import repeat
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns
from tqdm import tqdm

from molpal.acquirer import metrics
from utils import (
    extract_smis, build_true_dict, chunk, read_scores,
    style_axis, abbreviate_k_or_M
)

sns.set_theme(style='white', context='paper')

def gather_experiment_predss(experiment) -> Tuple[List, List]:
    chkpts_dir = Path(experiment) / 'chkpts'

    chkpt_iter_dirs = sorted(
        chkpts_dir.iterdir(), key=lambda p: int(p.stem.split('_')[-1])
    )[1:]
    try:                         # new way
        preds_npzs = [np.load(chkpt_iter_dir / 'preds.npz')
                      for chkpt_iter_dir in chkpt_iter_dirs]
        predss, varss = zip(*[
            (preds_npz['Y_pred'], preds_npz['Y_var'])
            for preds_npz in preds_npzs
        ])

        return predss, varss
    except FileNotFoundError:   # old way
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

    # import pdb; pdb.set_trace()
    for Y_pred, Y_var, new_points in zip(Y_preds, Y_vars, new_points_by_epoch):
        explored.update(new_points)
        ys = list(explored.values())
        Y = np.nan_to_num(np.array(ys, dtype=float), nan=-np.inf)
        current_max = np.partition(Y, -k)[-k]

        Us.append(metrics.calc(
            metric, Y_pred, Y_var,
            current_max, 0., beta, xi, False
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

def plot_reward_curves(yss: Iterable[Iterable[np.ndarray]],
                       labels: Iterable[str], bs: int, title: str):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 4))
    fmt = '-'

    y_min, y_max = 0, 0
    for ys, label in zip(yss, labels):
        Y = np.stack(ys)
        y_mean = Y.mean(axis=0)
        y_sd = Y.std(axis=0)

        y_max = max(y_max, max(y_mean))
        y_min = max(y_min, max(y_mean))

        x = np.arange(len(y_mean)) + 1
        ax.plot(x, y_mean, fmt, label=label, alpha=0.9)
        if len(Y) >= 3:
            ax.fill_between(x, y_mean-y_sd, y_mean+y_sd, alpha=0.3)

    n_iters = len(x) // bs
    ax.vlines([bs * (i+1) for i in range(n_iters)], y_min, y_max,
              color='r', ls='dashed', lw=0.5)
    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    ax.xaxis.set_major_formatter(formatter)
    style_axis(ax)
    ax.set_ylabel(title)
    ax.legend(loc='lower right')

    fig.tight_layout()

    return fig

#-----------------------------------------------------------------------------#

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
    parser.add_argument('-m', '--metrics', nargs='+', default=repeat('greedy'),
                        help='the respective acquisition metric used for each experiment')
    parser.add_argument('-k', type=int,
                        help='')
    parser.add_argument('-N', type=int,
                        help='the number of top scores from which to calculate the reward')
    parser.add_argument('-r', '--reward',
                        choices=('scores', 'smis', 'top-k-ave', 'total-ave'),
                        help='the type of reward to calculate')
    parser.add_argument('--split', type=float, default=0.004,
                        help='the split size to plot when using model-metrics mode')
    parser.add_argument('--reps', type=int, nargs='+',
                        help='the number of reps for each configuration. I.e., you passed in the arguments: --expts e1_a e1_b e1_c e2_a e2_b where there are three reps of the first configuration and two reps of the seecond. In this case, you should pass in: --reps 3 2. By default, the program assumed each experiment is a unique configuration.')
    parser.add_argument('--labels', nargs='+',
                        help='the label of each trace on the plot. Will use the metric labels if not specified. NOTE: the labels correspond the number of different configurations. I.e., if you pass in the args: --expts e1_a e1_b e1_c --reps 3, you only need to specify one label: --labels l1')
    parser.add_argument('--name',
                        help='the filepath to which the plot should be saved')

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

    reward_curvess = chunk(reward_curves, args.reps or [])

    bs = int(args.split * len(smis))
    title = {
        'smis': f'Percentage of Top-{args.N} SMILES Found',
        'scores': f'Percentage of Top-{args.N} scores Found',
        'top-k-ave': f'Top-{args.N} average',
        'total-ave': f'Total average'
    }[args.reward]

    fig = plot_reward_curves(
        reward_curvess, args.labels or args.metrics, bs, title
    ).savefig(args.name)

    exit()