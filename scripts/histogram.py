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

if __name__ == '__main__':
    # data_path = sys.argv[1]
    # score_col = sys.argv[2]
    # name = sys.argv[3]
    # k = int(sys.argv[4])
    # write_histogram(data_path, score_col, name, k, clip_positive)

    DATA_PATH = '/n/shakfs1/users/dgraff/data'
    paths = [
        f'{DATA_PATH}/4UNN_Enamine10k_scores.csv',
        f'{DATA_PATH}/4UNN_Enamine50k_scores.csv',
        f'{DATA_PATH}/4UNN_EnamineHTS_scores.csv',
        f'{DATA_PATH}/AmpC_100M_scores.csv.gz',
    ]
    score_cols = [2, 2, 1, 2]
    names = ['10k', '50k', 'HTS', 'AmpC']
    ks = [100, 500, 1000, 50000]

    for path, score_col, name, k in zip(paths, score_cols, names, ks):
        scores = extract_scores(path, score_col)
        if name in ['10k', '50k', 'HTS']:
            clip_positive = True
        else:
            clip_positive = False
        write_histogram(path, score_col, name, k, clip_positive)
