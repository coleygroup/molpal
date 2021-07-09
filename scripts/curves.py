from argparse import ArgumentParser
from collections import Counter
import csv
from functools import partial
import gzip
import heapq
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns
from tqdm import tqdm, trange

REPS = 3

sns.set_theme(style='white', context='paper')

palette = []
for cmap in ('Purples', 'Reds', 'Greens', 'Blues', 'Oranges'):
    palette.extend(sns.color_palette(cmap, REPS))

sns.set_palette(palette)

def gather_experiment_predss(experiment) -> List[np.ndarray]:
    chkpts_dir = Path(experiment) / 'chkpts'

    chkpt_iter_dirs = sorted(
        chkpts_dir.iterdir(), key=lambda p: int(p.stem.split('_')[-1])
    )[1:]

    preds_npzs = [np.load(chkpt_iter_dir / 'preds.npz')
                  for chkpt_iter_dir in chkpt_iter_dirs]
    predss, varss = [
        (preds_npz['Y_mean'], preds_npz['Y_var'])
        for preds_npz in preds_npzs
    ]
    return predss, varss

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
        for row in tqdm(reader, desc='Getting smis', leave=False):
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
        for row in tqdm(reader, desc='Building dict', leave=False):
            try:
                d_smi_score[row[smiles_col]] = c * float(row[score_col])
            except ValueError:
                continue

    return d_smi_score

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

def get_new_smis_by_epoch(experiment) -> List[Dict]:
    """get the set of new points and associated scores acquired at each
    iteration in the list of scores_csvs that are already sorted by iteration"""
    data_dir = Path(experiment) / 'data'

    scores_csvs = [p for p in data_dir.iterdir() if 'final' not in p.stem]
    scores_csvs = sorted(
        scores_csvs, key=lambda p: int(p.stem.split('_')[-1])
    )

    all_smis = set()
    new_smiss = []
    for scores_csv in scores_csvs:
        scores, failures = read_scores(scores_csv)
        scores.update(failures)
        new_smis = {smi: score for smi, score in scores.items()
                    if smi not in all_smis}
        new_smiss.append(new_smis)
        all_smis.update(new_smis)
    
    return new_smiss

def calculate_reward_in_order(
        experiment: str, metric: str,
        d_smi_idx: List[str], true_top_k: List,
        reward: str = 'scores'
    ) -> np.ndarray:
    """calculate the ordering of all hits found in a run of molpal

    I.e., if molpal samples 10 molecules and 6 of them are hits, it selected
    those 10 molecules with a specific and thus discovered hits in that given
    ordering as well

    Parameters
    ----------
    experiment : str
        the output directory of a molpal run
    metric : str
        the acquisition metric used
    smis : List[str]
        the library/pool of SMILES strings used for that run (in the same 
        ordering)
    true_top_k : List
        the list of the true top-k molecules as tuples of their SMILES string
        and associated score
    k : int
        [description]
    reward : str, default='scores'
        [description], by 
    maximize : bool, default=False
        [description], by 

    Returns
    -------
    np.ndarray    """
    predss, varss = gather_experiment_predss(experiment)
    Us = predss # fix with other metrics later

    k = len(true_top_k)
    all_points_in_order = []

    new_smis_by_epoch = get_new_smis_by_epoch(experiment)
    all_points_in_order.extend(new_smis_by_epoch[0].items())

    for new_points, U in zip(new_smis_by_epoch[1:], Us):
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
    elif reward == 'average':
        # import pdb; pdb.set_trace()
        reward_curve = np.zeros(len(all_points_in_order), dtype='f8')
        heap = []
        for i, (_, score) in enumerate(all_points_in_order[:k]):
            if score is not None:
                heapq.heappush(heap, score)
            top_k_avg = sum(heap) / k
            reward_curve[i] = top_k_avg
        # print(heap)
        reward_curve[:k] = top_k_avg
        for i, (_, score) in enumerate(all_points_in_order[k:]):
            if score is not None:
                heapq.heappushpop(heap, score)

            top_k_avg = sum(heap) / k
            reward_curve[i+k] = top_k_avg
        # for i in trange(k, len(all_points_in_order), desc='calculating'):
        #     _, current_scores = zip(*all_points_in_order[:i+1])
        #     top_k_scores = sorted(current_scores, reverse=True)[:k]
        #     top_k_avg = sum(top_k_scores) / k
        #     reward_curve[i] = top_k_avg
    else:
        raise ValueError

    return reward_curve

def plot_reward_curve(reward_curves: Iterable[np.ndarray],
                      metrics: Iterable[str], k: int, reward: str, bs: int):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 4))
    fmt = '-'

    y_max = 0
    for reward_curve, metric in zip(reward_curves, metrics):
        # Y = 100 * np.cumsum(hits_in_order) / k
        Y = reward_curve
        # print(Y)
        # print(np.where(Y==0))
        y_max = max(y_max, max(reward_curve))
        X = np.arange(len(reward_curve)) + 1
        ax.plot(X, reward_curve, fmt, label=metric, alpha=0.9)

    n_iters = len(X) // bs
    # ax.vlines([bs * (i+1) for i in range(n_iters)], 0, y_max,
    #           color='r', ls='dashed', lw=0.5)
    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    ax.xaxis.set_major_formatter(formatter)
    style_axis(ax)
    ax.set_ylabel(f'Percentage of Top-{k} {reward.capitalize()} Found')
    
    # ax.legend(loc='upper left', title='Metric')

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
    parser.add_argument('-m', '--metrics', nargs='+',
                        help='the acquisition metric used')
    parser.add_argument('-k', type=int,
                        help='the number of top scores from which to calculate performance')
    parser.add_argument('-r', '--reward',
                        help='the type of reward to calculate')
    parser.add_argument('--split', type=float, default=0.004,
                        help='the split size to plot when using model-metrics mode')
    parser.add_argument('--name')
    
    args = parser.parse_args()
    args.title_line = not args.no_title_line

    smis = extract_smis(args.library, args.smiles_col, args.title_line)
    d_smi_score = build_true_dict(
        args.true_csv, args.smiles_col, args.score_col,
        args.title_line, args.maximize
    )

    true_smis_scores = sorted(d_smi_score.items(),
                              key=lambda kv: kv[1])[::-1]
    true_top_k = true_smis_scores[:args.k]
    d_smi_idx = {smi: i for i, smi in tqdm(enumerate(smis),
                                           desc='indexing', leave=False)}
    reward_curves = []
    for experiment, metric in zip(args.experiments, args.metrics):
        reward_curves.append(calculate_reward_in_order(
            experiment, metric, d_smi_idx, true_top_k, args.reward
        ))

    bs = int(args.split * len(smis))
    fig = plot_reward_curve(
        reward_curves, args.metrics, args.k, args.reward, bs
    )

    fig.savefig(args.name)

    exit()