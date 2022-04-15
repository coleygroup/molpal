import datetime
import os
import signal
import sys
from telnetlib import XDISPLOC
import ray 
from molpal import args, Explorer, featurizer, pools, acquirer, models
import numpy as np 
import heapq
from pathlib import Path
import csv 
from molpal.models.nnmodels import (
        NNModel, NNDropoutModel, NNEnsembleModel, NNTwoOutputModel
    )
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygmo as pg
import imageio

# To start: in vscode, check interpreter in conda environment 'molpalmoo'
# kill current termial 
# go to menu bar > terminal > new terminal 
# type in ' ray start --head  ' (include --num-gpus N --num-cpus N if desired)

def clean_update_scores(scores, new_scores):
    for x, y in new_scores.items():
        if y is None: 
            print(x)
            print('Above Molecule Failed')
        else:
            scores[x] = y

def read_data(path, smiles_col, data_cols):
    reader = csv.reader(open(Path(path)))
    data = {}
    for row in reader: 
        try: 
            key = row[smiles_col]
            val = []
            for col in data_cols:
                val.append(float(row[col]))
            data[key]=val
        except:
            pass
    return data

def plot_parity(iter_storage, pool, data, Y_pred, n):
    smiles_list = list(pool.smis())
    Y_pred_dict = {smiles_list[i]: Y_pred[i] for i in range(len(smiles_list))}
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(np.array(list(data.values()))[:,n],Y_pred, color='gray',label='Not batched')
    ax.scatter([],[],color='blue', label='batched')
    for smiles in iter_storage['smis']:
        ax.scatter(data[smiles][n],Y_pred_dict[smiles],color='blue')

    ax.plot([-5,-11],[-5,-11],color='k')
    ax.set_xlabel('True Docking Score')
    ax.set_ylabel('Predicted Docking Score')
    ax.legend()
    ax.show()

def plot_acquired(data, scores):
    true_scores = np.array(list(data.values()))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(true_scores[:,0], true_scores[:,1], color='gray',label='Not sampled')
    ax.scatter([],[],color='blue', label='sampled')
    for smiles in scores:
        ax.scatter(-scores[smiles][0], scores[smiles][1], color='blue')
    
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.legend()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def retrain_models(storage, scores):
    scorelist = np.array(list(scores.items()))
    xs = scorelist[:,0]
    ys = np.vstack(scorelist[:,1])
    y0s = ys[:,0]
    y1s = ys[:,1]
    storage['model0'] = NNModel(input_size=kwargs['length'], num_tasks=1, \
                    test_batch_size=100,  # so one batch wouldn't have like 10 molecules 
                    layer_sizes=None, 
                    dropout=None,
                    activation='relu',
                    uncertainty=None,
                    model_seed=None) 
    storage['model1'] = NNModel(input_size=kwargs['length'], num_tasks=1, \
                    test_batch_size=100,  
                    layer_sizes=None, 
                    dropout=None,
                    activation='relu',
                    uncertainty=None,
                    model_seed=None) 
    storage['model0'].train(xs, y0s,
                featurizer=storage['featurizer'], retrain = True)

    storage['model1'].train(xs, y1s,
                featurizer=storage['featurizer'], retrain = True)

def acquire_ndf(Y0_pred, Y1_pred, storage, iter_string, scores):
    ndf, _, _, _ = pg.fast_non_dominated_sorting(np.vstack((-Y0_pred, Y1_pred)).T)
    ndf = np.concatenate(ndf, axis=0)
    new_points = []

    smiles_list = list(storage['pool'].smis())
    i = 0
    while len(new_points) < 100:
        new_smile = smiles_list[ndf[i]]
        if new_smile in scores: 
            i = i + 1
        else:
            new_points.append(ndf[i]) 
            i = i + 1

    storage[iter_string] = {}
    storage[iter_string]['smis'] = [smiles_list[i] for i in new_points]

    new_scores = {}
    for new in storage[iter_string]['smis']:
        new_scores[new] = data[new]
    return new_scores

def predict(storage):
    Y0_pred, _ = storage['model0'].apply(
            x_ids=storage['pool'].smis(), x_feats=storage['pool'].fps(), 
            batched_size=None, size=len(storage['pool']), 
            mean_only=True
        )
    Y1_pred, _ = storage['model1'].apply(
            x_ids=storage['pool'].smis(), x_feats=storage['pool'].fps(), 
            batched_size=None, size=len(storage['pool']), 
            mean_only=True
        )
    return Y0_pred, Y1_pred

data = read_data('data/selectivity_data.csv', 0, [1,2])
# shutdown any previous ray cluster 
ray.shutdown()

# set up ray cluster 
ray.init(num_cpus=2, num_gpus=0)
print(ray.cluster_resources())

# setup run
arguments = '-o lookup -l libraries/selectivity_data.csv --output-dir selectivity -vvv \
--objective-config examples/objective/selectivity.ini'.split()
params = vars(args.gen_args(arguments))
params['models'] = ['rf', 'rf']
params['metric'] = 'greedy'
path = params.pop("output_dir")
kwargs = params
# explorer = Explorer(path, **params)

# print(explorer)


print('Starting Exploration ...')
storage = {}
storage['featurizer'] = featurizer.Featurizer(fingerprint=kwargs['fingerprint'],
            radius=kwargs['radius'], length=kwargs['length']
        )
storage['pool'] = pools.pool(featurizer=storage['featurizer'],\
    **kwargs)
storage['Acquirer'] = acquirer.Acquirer(size=len(storage['pool']),**kwargs)
xs = storage['pool'].smis()
storage['scores'] = {}

# first iteration

# select random points, first iteration
U = acquirer.metrics.random(np.empty(storage['Acquirer'].size))

heap = []
for x, u in zip(xs, U):
    if len(heap) < storage['Acquirer'].init_size:
        heapq.heappush(heap, (u, x))
    else:
        heapq.heappushpop(heap, (u, x))

storage['iter1'] = {}
storage['iter1']['smis'] = [x for _, x in heap]
# collect new scores, update scores
scores = {}
new_scores = {}
for new in storage['iter1']['smis']:
    new_scores[new] = data[new]

clean_update_scores(scores, new_scores)


# retrain new model from scratch 

retrain_models(storage, scores)

Y0_pred, Y1_pred = predict(storage)
images = [plot_acquired(data, storage['iter1'])]
# plot_parity(storage['iter1'], storage['pool'], data, Y0_pred, 0)
# plot_parity(storage['iter1'], storage['pool'], data, Y1_pred, 1)

for i in range(2,11):
    iterlabel = 'iter' + str(i)
    new_scores = acquire_ndf(Y0_pred, Y1_pred, storage, iterlabel, scores)
    clean_update_scores(scores, new_scores)
    retrain_models(storage, scores)
    Y0_pred, Y1_pred = predict(storage) 
    images.append(plot_acquired(data, scores))
    print(len(scores))

imageio.mimsave('./powers.gif', images)