import argparse
from collections import Counter
import csv
from functools import partial
import gzip
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

sns.set_theme(style='white', context='paper')

def extract_scores(scores_csv, score_col=1, title_line=True):
    if Path(scores_csv).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open
    
    with open_(scores_csv) as fid:
        reader = csv.reader(fid)
        if title_line:
            next(reader)
        scores = []
        for row in tqdm(reader):
            try:
                score = float(row[score_col])
            except ValueError:
                continue
            scores.append(score)
    return np.sort(np.array(scores))

def write_histogram(path, score_col, name, k, clip_positive=False):
    scores = extract_scores(path, score_col)
    if clip_positive:
        scores = scores[scores < -0.1]
    cutoff = scores[k]
    fig = plt.figure(figsize=(10, 4))

    BINWIDTH = 0.1
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharex=ax1)
    for ax in (ax1, ax2):
        ax.hist(scores, color='b', edgecolor='none',
                bins=np.arange(min(scores), max(scores)+BINWIDTH, BINWIDTH))
        ax.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
        ax.set_ylabel('Count')
        ax.grid(True, linewidth=1, color='whitesmoke')
    ax2.set_yscale('log')
    ax2.set_ylabel('Count')
    
    # add global x-axis label
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none',
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Score')

    plt.tight_layout()
    plt.savefig(f'{name}_score_histogram.pdf')
    plt.clf()

parser = argparse.ArgumentParser()
parser.add_argument('--paths', nargs='+',
                    help='the paths containing the datasets')
parser.add_argument('--names', nargs='+',
                    help='the name of each dataset')
parser.add_argument('--score-cols', nargs=+, type=int,
                    help='the column in each dataset CSV containing the score')
parser.add_argument('--top-ks', nargs=+, type=int,
                    help='the value of k to use for each dataset')
parser.add_argument('--clip-positive', action='store_true', default=False,
                    help='whether to clip values >= 0 from the datasets.')
                    
if __name__ == '__main__':
    args = parser.parse_args()
    for path, score_col, name, k in zip(args.paths, args.score_cols
                                        args.names, args.top_ks):
        scores = extract_scores(path, score_col)
        write_histogram(path, score_col, name, k, args.clip_positive)
