import csv
from operator import itemgetter
from pathlib import Path
import pickle
import sys
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io

import plotly.io._orca
import retrying
unwrapped = plotly.io._orca.request_image_with_retrying.__wrapped__
wrapped = retrying.retry(wait_random_min=1000)(unwrapped)
plotly.io._orca.request_image_with_retrying = wrapped
plotly.io.templates.default = 'plotly_white'

def read_scores(scores_csv: str) -> Dict:
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

def get_new_points_by_epoch(scores_csvs: List[str]) -> List[Dict]:
    """get the set of new points and associated scores acquired at each
    iteration in the list of scores_csvs that are already sorted by iteration"""
    all_points = dict()
    new_points_by_epoch = []
    for scores_csv in scores_csvs:
        scores, _ = read_scores(scores_csv)
        new_points = {smi: score for smi, score in scores.items()
                      if smi not in all_points}

        new_points_by_epoch.append(new_points)
        all_points.update(new_points)
    
    return new_points_by_epoch

# def make_umap_plot_series(parent_dirs, models):
#     scores_csvss = [
#         [p_csv for p_csv in Path(parent_dir).iterdir() if 'iter' in p_csv.stem]
#         for parent_dir in parent_dirs
#     ]
#     scores_csvss = [
#         sorted(scores_csvs, key=lambda p: int(p.stem.split('_')[4]))
#         for scores_csvs in scores_csvss
#     ]    
#     iters = [0, 1, 3, 5]
    
#     australia_coords = dict(x0=4.6, x1=7.5, y0=-6.6, y1=-5.4)
#     japan_coords = dict(x0=15.2, x1=16.9, y0=3.2, y1=5.8)
    
#     umap_series = make_subplots(
#         rows=len(scores_csvss)+2, cols=len(iters),
#         shared_xaxes=True, shared_yaxes=True,
#         specs=[[{"rowspan": 2, "colspan": 2}, None,
#                 {"rowspan": 2, "colspan": 2}, None],
#                [None, None, None, None],
#                [{}, {}, {}, {}],
#                [{}, {}, {}, {}],
#                [{}, {}, {}, {}]],
#         x_title='Iteration'
#     )
    
#     shapes = []
#     def add_series_row(umap_series, scores_csvs, row, iters, model):
#         col = 1
#         for j, new_points in enumerate(get_new_points_by_epoch(scores_csvs)):
#             N = 3+4*(row-3)+(col-1)
#             if j not in iters:
#                 continue

#             smis, scores = zip(*new_points.items())
#             idxs = [d_smi_idx[smi] for smi in smis]
#             new_points_embedding = go.Scatter(
#                 x=fps_embedded[idxs, 0], y=fps_embedded[idxs, 1],
#                 mode='markers',
#                 marker=dict(size=3, color=scores, coloraxis='coloraxis2')
#             )
#             umap_series.add_trace(new_points_embedding, row, col)
#             umap_series.update_yaxes(
#                 row=row, col=col, title=(model if col==1 else None))
#             umap_series.update_xaxes(
#                 row=row, col=col, title=(j if row==5 else None))

#             shapes.append(dict(
#                 type='circle', line_color='black',
#                 xref=f'x{N}', yref=f'y{N}',
#                 **australia_coords,
#             ))
#             shapes.append(dict(
#                 type='circle', line_color='black',
#                 xref=f'x{N}', yref=f'y{N}',
#                 **japan_coords,
#             ))
#             col += 1

#         return umap_series
    
#     density_plot = go.Histogram2d(
#         x=fps_embedded[:, 0],
#         y=fps_embedded[:, 1],
#         coloraxis='coloraxis1'
#     )
    
#     top_1k_idxs = [d_smi_idx[smi] for smi in true_top_1k_smis] 
#     top_1k_fps_embedded = fps_embedded[top_1k_idxs, :]
#     top_1k_embedding = go.Scatter(
#         x=top_1k_fps_embedded[:, 0],
#         y=top_1k_fps_embedded[:, 1],
#         mode='markers', marker=dict(size=3, color='grey')
#     )
    
#     umap_series.add_trace(top_1k_embedding, row=1, col=1)
#     shapes.append(dict(
#         type='circle', line_color='black',
#         xref=f'x2', yref=f'y2',
#         **australia_coords
#     ))
#     shapes.append(dict(
#         type='circle', line_color='black',
#         xref=f'x2', yref=f'y2',
#         **japan_coords
#     ))
#     umap_series.add_trace(density_plot, row=1, col=3)
#     shapes.append(dict(
#         type='circle', line_color='black',
#         xref=f'x1', yref=f'y1',
#         **australia_coords,
#     ))
#     shapes.append(dict(
#         type='circle', line_color='black',
#         xref=f'x1', yref=f'y1',
#         **japan_coords,
#     ))
    
#     for i, scores_csvs_model in enumerate(zip(scores_csvss, models)):
#         scores_csvs, model = scores_csvs_model
#         add_series_row(umap_series, scores_csvs, 2+(i+1), iters, model)
    
#     # add borders
#     border_params = dict(
#         showgrid=False, zeroline=False,
#         visible=True, mirror=True, linewidth=2,
#         linecolor='black'
#     )
#     for col in [1, 3]:
#         umap_series.update_xaxes(row=1, col=col, **border_params)
#         umap_series.update_yaxes(row=1, col=col, **border_params)
#     for row in range(3, 6):
#         for col in range(1, 5):
#             umap_series.update_xaxes(row=row, col=col, **border_params)
#             umap_series.update_yaxes(row=row, col=col, **border_params)
    
#     SIZE = 250
#     HEIGHT = SIZE*(len(scores_csvss)+2)
#     WIDTH = SIZE*len(iters)
    
#     for i in umap_series['layout']['annotations']:
#         i['font'] = dict(color='black', size=20, family='sans-serif')

#     umap_series.update_layout(
#         height=HEIGHT, width=WIDTH, showlegend=False,
#         coloraxis1=dict(colorscale='purples',
#                         colorbar=dict(
#                             title='Points', len=2/5,
#                             y=1.02, yanchor='top')),
#         coloraxis2=dict(colorscale='plasma', cmin=zmin, cmax=zmax,
#                         colorbar=dict(
#                             title='Score', len=3/5,
#                             y=0, yanchor='bottom')),
#         shapes=shapes,
#         font=dict(color='black', size=18, family='sans-serif')
#     )
#     return umap_series

scores_dict_pkl = sys.argv[1] # '/n/shakfs1/users/dgraff/data/4UNN_EnamineHTS_scores_dict.pkl'
smis_csv = sys.argv[2] #'/n/shakfs1/users/dgraff/libraries/EnamineHTS.csv'
fps_embedded_npy = sys.argv[3] # 'EnamineHTS_pair_umap.npy'
parent_data_dir = sys.argv[4] # '/n/shakfs1/users/dgraff/HTS_retrain/001'

d_smi_score = pickle.load(open(scores_dict_pkl, 'rb'))
true_top_1k = dict(sorted(d_smi_score.items(), key=itemgetter(1))[:1000])
true_top_1k_smis = set(true_top_1k.keys())

with open(smis_csv, 'r') as fid:
    reader = csv.reader(fid); next(reader)
    smis = [row[0] for row in tqdm(reader)]
d_smi_idx = {smi: i for i, smi in enumerate(smis)}
Z = np.array([-d_smi_score[smi] if smi in d_smi_score else 0
              for smi in tqdm(smis)])

zmin = -max(score for score in d_smi_score.values() if score < 0)
zmax = -min(d_smi_score.values())

fps_embedded = np.load(fps_embedded_npy)

parent_dirs = [
    f'{parent_data_dir}/HTS_rf_greedy_001_0_retrain/data',
    f'{parent_data_dir}/HTS_nn_greedy_001_0_retrain/data',
    f'{parent_data_dir}/HTS_mpn_greedy_001_0_retrain/data'
]
scores_csvss = [
    [p_csv for p_csv in Path(parent_dir).iterdir() if 'iter' in p_csv.stem]
    for parent_dir in parent_dirs
]
scores_csvss = [
    sorted(scores_csvs, key=lambda p: int(p.stem.split('_')[4]))
    for scores_csvs in scores_csvss
]

MODELS = ['RF', 'NN', 'MPN']
ITERS = [0, 1, 3, 5]

australia_coords = dict(x0=4.6, x1=7.5, y0=-6.6, y1=-5.4)
japan_coords = dict(x0=15.2, x1=16.9, y0=3.2, y1=5.8)

umap_figure = make_subplots(
    rows=len(scores_csvss)+2, cols=len(ITERS),
    shared_xaxes=True, shared_yaxes=True,
    specs=[[{"rowspan": 2, "colspan": 2}, None,
            {"rowspan": 2, "colspan": 2}, None],
            [None, None, None, None],
            [{}, {}, {}, {}],
            [{}, {}, {}, {}],
            [{}, {}, {}, {}]],
    x_title='Iteration'
)

shapes = []
def add_series_row(umap_series, scores_csvs, row, iters, model):
    col = 1
    for j, new_points in enumerate(get_new_points_by_epoch(scores_csvs)):
        N = 3+4*(row-3)+(col-1)
        if j not in iters:
            continue

        smis, scores = zip(*new_points.items())
        idxs = [d_smi_idx[smi] for smi in smis]
        new_points_embedding = go.Scatter(
            x=fps_embedded[idxs, 0], y=fps_embedded[idxs, 1],
            mode='markers',
            marker=dict(size=3, color=scores, coloraxis='coloraxis2')
        )
        umap_figure.add_trace(new_points_embedding, row, col)
        umap_figure.update_yaxes(
            row=row, col=col, title=(model if col==1 else None))
        umap_figure.update_xaxes(
            row=row, col=col, title=(j if row==5 else None))

        shapes.append(dict(
            type='circle', line_color='black',
            xref=f'x{N}', yref=f'y{N}',
            **australia_coords,
        ))
        shapes.append(dict(
            type='circle', line_color='black',
            xref=f'x{N}', yref=f'y{N}',
            **japan_coords,
        ))
        col += 1

    return umap_series

density_plot = go.Histogram2d(
    x=fps_embedded[:, 0],
    y=fps_embedded[:, 1],
    coloraxis='coloraxis1'
)

top_1k_idxs = [d_smi_idx[smi] for smi in true_top_1k_smis] 
top_1k_fps_embedded = fps_embedded[top_1k_idxs, :]
top_1k_embedding = go.Scatter(
    x=top_1k_fps_embedded[:, 0],
    y=top_1k_fps_embedded[:, 1],
    mode='markers', marker=dict(size=3, color='grey')
)

umap_figure.add_trace(top_1k_embedding, row=1, col=1)
shapes.append(dict(
    type='circle', line_color='black',
    xref=f'x2', yref=f'y2',
    **australia_coords
))
shapes.append(dict(
    type='circle', line_color='black',
    xref=f'x2', yref=f'y2',
    **japan_coords
))
umap_figure.add_trace(density_plot, row=1, col=3)
shapes.append(dict(
    type='circle', line_color='black',
    xref=f'x1', yref=f'y1',
    **australia_coords,
))
shapes.append(dict(
    type='circle', line_color='black',
    xref=f'x1', yref=f'y1',
    **japan_coords,
))

for i, (scores_csvs, model) in enumerate(zip(scores_csvss, MODELS)):
    # add_series_row(umap_figure, scores_csvs, 2+(i+1), iters, model)
    for j, new_points in enumerate(get_new_points_by_epoch(scores_csvs)):
        if j not in ITERS:
            continue

        row = 2 + (i + 1)
        col = j+1
        smis, scores = zip(*new_points.items())
        idxs = [d_smi_idx[smi] for smi in smis]
        new_points_embedding = go.Scatter(
            x=fps_embedded[idxs, 0], y=fps_embedded[idxs, 1],
            mode='markers',
            marker=dict(size=3, color=scores, coloraxis='coloraxis2')
        )
        umap_figure.add_trace(new_points_embedding, row, col)
        umap_figure.update_yaxes(
            row=row, col=col, title=(model if col==1 else None))
        umap_figure.update_xaxes(
            row=row, col=col, title=(j if row==5 else None))

        N = 3+4*(row-3)+(col-1)
        shapes.append(dict(
            type='circle', line_color='black',
            xref=f'x{N}', yref=f'y{N}',
            **australia_coords,
        ))
        shapes.append(dict(
            type='circle', line_color='black',
            xref=f'x{N}', yref=f'y{N}',
            **japan_coords,
        ))
        col += 1

# add borders
border_params = dict(
    showgrid=False, zeroline=False,
    visible=True, mirror=True, linewidth=2,
    linecolor='black'
)
for col in [1, 3]:
    umap_figure.update_xaxes(row=1, col=col, **border_params)
    umap_figure.update_yaxes(row=1, col=col, **border_params)
for row in range(3, 6):
    for col in range(1, 5):
        umap_figure.update_xaxes(row=row, col=col, **border_params)
        umap_figure.update_yaxes(row=row, col=col, **border_params)

SIZE = 250
HEIGHT = SIZE*(len(scores_csvss)+2)
WIDTH = SIZE*len(ITERS)

for i in umap_figure['layout']['annotations']:
    i['font'] = dict(color='black', size=20, family='sans-serif')

umap_figure.update_layout(
    height=HEIGHT, width=WIDTH, showlegend=False,
    coloraxis1=dict(colorscale='purples',
                    colorbar=dict(
                        title='Points', len=2/5,
                        y=1.02, yanchor='top')),
    coloraxis2=dict(colorscale='plasma', cmin=zmin, cmax=zmax,
                    colorbar=dict(
                        title='Score', len=3/5,
                        y=0, yanchor='bottom')),
    shapes=shapes,
    font=dict(color='black', size=18, family='sans-serif')
)

umap_figure.write_image('umap_figure.pdf')
