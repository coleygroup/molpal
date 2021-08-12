from argparse import ArgumentParser
import heapq
from itertools import repeat
from typing import Iterable

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

from experiment import Experiment
from utils import (
    extract_smis, build_true_dict, chunk, style_axis, abbreviate_k_or_M
)

sns.set_theme(style='white', context='paper')

def plot_regret(
        total_avgss: Iterable[Iterable[np.ndarray]], init_sizes: Iterable[int],
        reward: str, true_scores: Iterable[float], k: int,
        labels: Iterable[str], true_size: int
    ):

    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)

    true_scores = np.concatenate((
        -np.array(true_scores), np.zeros(true_size - len(true_scores))
    ))
    x = np.arange(len(true_scores)) + 1

    if reward == 'total-ave':
        y_true = np.cumsum(np.sort(true_scores)) / x
    elif reward == 'top-k-ave':
        y_true = np.empty(len(true_scores))
        y_true[:] = np.sort(true_scores)[:k].mean()
    else:
        raise ValueError

    ax0.plot(x, y_true, '--', c='k', lw=2, label='True')

    for ys, init_size, label in zip(total_avgss, init_sizes, labels):
        Y = -np.stack(ys)[:, init_size:] # hack b/c i know true data is neg
        y_mean = Y.mean(axis=0)
        y_sd = Y.std(axis=0)

        x = np.arange(len(y_mean)) + 1 + init_size

        R = Y - y_true[x-1]
        r_mean = R.mean(axis=0)
        r_sd = R.std(axis=0)

        ax0.plot(x, y_mean, alpha=0.7, label=label)
        ax1.plot(x, r_mean, alpha=0.7)
        ax2.plot(x, np.cumsum(r_mean), alpha=0.7)

        if len(Y) >= 3:
            ax0.fill_between(x, y_mean-y_sd, y_mean+y_sd, alpha=0.3)
            ax1.fill_between(x, r_mean-r_sd, r_mean+r_sd, alpha=0.3)
            ax2.fill_between(
                x, np.cumsum(r_mean-r_sd),
                np.cumsum(r_mean+r_sd), alpha=0.3
            )

    random_scores = -np.random.choice(true_scores, size=len(x)+init_size)
    if reward == 'total-ave':
        x_rand = np.arange(len(random_scores)) + 1
        y_rand = np.cumsum(random_scores) / x_rand
        regret_rand = y_rand - y_true

        ax1.plot(x_rand, regret_rand, '--', lw=0.5, c='grey')
        ax2.plot(x_rand, np.cumsum(regret_rand), '--', lw=0.5, c='grey')
        
    elif reward == 'top-k-ave':
        y_rand = np.empty(len(random_scores))
        heap = []
        for score in random_scores[:k]:
            heapq.heappush(heap, score)
        y_rand[:k] = sum(heap) / k

        for i, score in enumerate(random_scores[k:]):
            heapq.heappushpop(heap, score)
            y_rand[i+k] = sum(heap) / k
    else:
        raise ValueError(f'unrecognized reward: "{reward}"')

    ax0.plot(x, -y_rand[x-1], '--', c='grey', lw=0.5, label='random')

    formatter = ticker.FuncFormatter(abbreviate_k_or_M)
    for ax in (ax0, ax1, ax2):
        ax.xaxis.set_major_formatter(formatter)
        style_axis(ax)

    ax0.set_ylabel('Total average')
    ax0.set_xlim(right=4*len(y_mean))
    ax0.legend(loc='lower right')

    ax1.set_ylabel('Regret')
    # ax1.set_ylim(bottom=0)

    ax2.set_ylabel('Cumulative Regret')
    # ax2.set_ylim(bottom=0)

    return fig

#-----------------------------------------------------------------------------#

def plot_estimates(true_scores: Iterable[float], true_size: int):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(4, 4), constrained_layout=True
    )

    true_scores = -np.array(true_scores)
    padding = np.zeros(true_size-len(true_scores))
    true_scores = np.concatenate((true_scores, padding))
    X = np.arange(len(true_scores)) + 1
    Y_true = np.cumsum(np.sort(true_scores)) / X

    ax1.plot(X, Y_true, '--', c='k', label='True', alpha=0.8)

    text = [f'True: ({Y_true[0]}, {Y_true[-1]:0.2f})']
    text2 = []

    ks = (100, 200, 1000, 2000, 4000, 8000)
    colors = sns.color_palette('Blues', len(ks))
    for i, k in tqdm(enumerate(ks), total=len(ks)):
        REPS = 5
        Y_samples = np.empty((REPS, true_size))
        E_as = np.empty(Y_samples.shape)
        for j in range(REPS):
            random_scores = np.random.choice(true_scores, size=k)
            mu, sd = norm.fit(random_scores)
            samples = norm.rvs(mu, sd, true_size)
            Y_samples[j] = np.cumsum(np.sort(samples)) / X
            E_as[j] = np.abs(Y_true - Y_samples[j])

        Y_mean = np.mean(Y_samples, axis=0)
        Y_std = np.std(Y_samples, axis=0)

        ax1.plot(X, Y_mean, label=k, alpha=0.8, c=colors[i])
        ax1.fill_between(X, Y_mean-Y_std, Y_mean+Y_std,
                        alpha=0.3, facecolor=colors[i])
        text.append(f'{k}: ({Y_mean[0]:0.2f}, {Y_mean[-1]:0.2f})')            
        
        E_a_mean = np.mean(E_as, axis=0)[:len(X)//10]
        # E_a_std = np.mean(E_as, axis=0)[:len(X)//10]
        ax2.plot(X[:len(X)//10], E_a_mean, c=colors[i])
        # ax2.fill_between(X[:len(X)//10], E_a_mean-E_a_std, E_a_mean+E_a_std,
        #                 alpha=0.3, facecolor=colors[i])
        text2.append(f'{k}: MAE: {E_a_mean.mean():0.2f}')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text = '\n'.join(text)
    ax1.text(
        0.75, 0.05, text, transform=ax1.transAxes, fontsize=4,
        verticalalignment='bottom', horizontalalignment='right', bbox=props
    )
    text2 = '\n'.join(text2)
    ax2.text(
        0.9, 0.9, text2, transform=ax2.transAxes, fontsize=4,
        verticalalignment='top', horizontalalignment='right', bbox=props
    )

    formatter = ticker.FuncFormatter(abbreviate_k_or_M)

    ax1.xaxis.set_major_formatter(formatter)
    ax1.set_ylabel('Rolling average')
    ax1.legend(loc='lower right', fontsize=6)

    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_ylabel('Absolute Error')

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
                        help='the acquisition metric used')
    parser.add_argument('-k', type=int,
                        help='the number of top scores from which to calculate performance')
    parser.add_argument('-r', '--regret',
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
    d_smi_idx = {smi: i for i, smi in enumerate(smis)}

    d_smi_score = build_true_dict(
        args.true_csv, args.smiles_col, args.score_col,
        args.title_line, args.maximize
    )
    true_smis_scores = sorted(d_smi_score.items(), key=lambda kv: kv[1])[::-1]
    true_top_k = true_smis_scores[:args.k]

    reward_curves = []
    init_sizes = []
    for experiment in args.experiments:
        experiment = Experiment(experiment, d_smi_idx)
        init_sizes.append(experiment.init_size)
        reward_curves.append(experiment.reward_curve(true_top_k, args.regret))

    reward_curvess = chunk(reward_curves, args.reps or [])
    init_sizes = [x[0] for x in chunk(init_sizes, args.reps or [])]

    _, true_scores = zip(*true_smis_scores)
    plot_regret(
        reward_curvess, init_sizes, args.regret,
        true_scores, args.k, args.labels, len(smis)
    ).savefig(args.name)

        # plot_estimates(true_scores, len(smis)).savefig(args.names[1], dpi=600)

    exit()