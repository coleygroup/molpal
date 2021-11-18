from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from itertools import islice
from operator import itemgetter
from pathlib import Path
import pickle
from typing import Dict, Iterable, List, Set, Tuple

from matplotlib.axes import Axes
from matplotlib import pyplot as plt, ticker
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from dataset import Dataset
from experiment import Experiment

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

def style_axis(ax: Axes):
    ax.set_xlabel(f'Molecules explored')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.xaxis.set_tick_params(rotation=30)
    ax.grid(True)

def abbreviate_k_or_M(x: float, pos) -> str:
    if x >= 1e6:
        return f'{x*1e-6:0.1f}M'
    if x >= 1e3:
        return f'{x*1e-3:0.0f}k'

    return f'{x:0.0f}'

def group_datasets(datasets: List[Dataset]) -> Dict:
    nested_dict = lambda: defaultdict(nested_dict)
    results = nested_dict()

    for dataset in datasets:
        results[str(dataset.split)][dataset.model][dataset.metric] = dataset

    return results


def plot_datasets(ax: Axes, datasets: List[Dataset], reward: str, label: str):
    for dataset in datasets:
        Y = dataset.get_reward(reward)
        ax.errorbar(dataset.num_acquired, Y[:, 0], yerr=Y[:, 1], label=label)
    

def plot_model_metrics(results: Dict, split: str, reward: str) -> Figure:
    fig, axs = plt.subplots(1, len(results[split]), sharex=True, sharey=True, figsize=(4/1.5*3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2

    for i, (model, ax) in enumerate(zip(results[split], axs)):
        for metric in results[split][model]:
            dataset = results[split][model][metric]
            Y = dataset.get_reward(reward)
            ax.errorbar(
                dataset.num_acquired,
                Y[:, 0],
                yerr=Y[:, 1],
                color=METRIC_COLORS[metric], 
                fmt=fmt,
                ms=ms,
                mec='black',
                capsize=capsize,
                label=metric
            )
        
        ys = [y*100 for y in ys]
        y_sds = [y*100 for y in y_sds]
        ax.plot(
            dataset.num_acquired,
            ys,
            fmt,
            ms=ms,
            color='grey',
            mec='black',
            capsize=capsize,
            label='random'
        )

        ax.set_title(model.upper())
        if i == 0:
            ax.set_ylabel(f'Percentage of Top-{dataset.N} {reward.capitalize()} Found')
            ax.legend(loc='upper left', title='Metric')
        
        style_axis(ax)

    return fig

def plot_results(results: Dict, primary: str, secondary: str):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(4/1.5 * 3, 4))

    fmt = 'o-'
    ms = 5
    capsize = 2

    primary_keys = {
        "SPLITS": SPLITS,
        "MODELS": MODELS,
        "METRICS": METRICS
    }[primary.upper()]

    secondary_keys = {
        "SPLITS": SPLITS,
        "MODELS": MODELS,
        "METRICS": METRICS
    }[secondary.upper()]

    for i, (k1, ax) in enumerate(zip(primary_keys, axs)):
        datasets = results[k1]
        for k2 in secondary_keys:
            pass