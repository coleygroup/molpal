import csv
from functools import partial
import gzip
from pathlib import Path

import numpy as np
import plotly.io
import plotly.graph_objects as go
from tqdm import tqdm

plotly.io.templates.default = 'plotly_white'

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
                score = round(float(row[score_col]), 1)
                if score >= 0:
                    continue
            except ValueError:
                continue
            scores.append(score)
    return scores

DATA_PATH = '/n/shakfs1/users/dgraff/data'
E10k = extract_scores(f'{DATA_PATH}/4UNN_Enamine10k_scores.csv', 2)
E50k = extract_scores(f'{DATA_PATH}/4UNN_Enamine50k_scores.csv', 2)
HTS = extract_scores(f'{DATA_PATH}/4UNN_EnamineHTS_scores.csv', 1)
AmpC = extract_scores(f'{DATA_PATH}/AmpC_100M_scores.csv.gz', 2)

libraries = ['Enamine 10k', 'Enamine 50k', 'Enamine HTS', 'AmpC']
cutoffs = [-9.5, -9.6, -11.0, -74.0]
for scores, library, cutoff in zip([E10k, E50k, HTS, AmpC], libraries, cutoffs):
    scores = np.array(scores)
    fig = go.Figure(data=[go.Histogram(x=scores)])
    fig.update_layout(
        title_text=f'Docking score distribution in {library} library',
        xaxis_title_text='Score', # xaxis label
        yaxis_title_text='Count', # yaxis label
    )
    fig.add_shape(type="line",
        xref="x", yref="y",
        x0=cutoff, x1=cutoff, y0=0, y1=0.01*len(scores),
        line=dict(
            color="red",
            width=1,
        ),
    )
    border_params = dict(
        visible=True, mirror=True, linewidth=2,
        linecolor='black'
    )
    fig.update_xaxes(range=[min(scores)-1, max(scores)+1], **border_params)
    fig.update_yaxes(**border_params)
    fig.write_image(f'{library}_score_histogram.pdf')