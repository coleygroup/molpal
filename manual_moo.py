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
import pandas as pd
import json

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
            data[key][0] = -data[key][0] # to make it negative
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

    ax.plot([5,11],[-5,-11],color='k')
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
        ax.scatter(scores[smiles][0], scores[smiles][1], color='blue')
    
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.legend()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def retrain_models(storage, scores):
    scorelist = np.array(list(scores.items()),dtype=object)
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
    ndf, _, _, _ = pg.fast_non_dominated_sorting(np.vstack((Y0_pred, Y1_pred)).T)
    ndf = np.concatenate(ndf, axis=0)
    new_points = []

    smiles_list = list(storage['pool'].smis())
    i = 0
    while len(new_points) < n_per_acq:
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

def plot_boxplot(acquired_scores):
    df0 = pd.DataFrame()
    df1 = pd.DataFrame()
    fig, ax = plt.subplots(2, 1,figsize=(7,5))
    for i in acquired_scores:
        np.array(list(data.values()))
        these_scores = np.array(list(acquired_scores[i].values()))
        y0 = these_scores[:,0]
        df0[str(i)] = y0
        y1 = these_scores[:,1]
        df1[str(i)] = y1
        x = np.random.normal(int(i), 0.04, these_scores.shape[0])
        ax[0].plot(x, y0, ms=1, marker="o", linestyle="None")
        ax[1].plot(x, y1, ms=1, marker="o", linestyle="None")
    df0.boxplot(ax=ax[0])
    df1.boxplot(ax=ax[1])
    ax[0].set_ylabel('objective 1 (minimize)')
    ax[1].set_ylabel('objective 2 (minimize)')
    ax[1].set_xlabel('iteration')

def calculate_hvs(storage,data):
    ref = np.max(np.array(list(data.values())),axis=0)
    hvs = np.zeros(n_iter)
    hv_all = pg.hypervolume(np.array(list(data.values())))
    scores_acq = np.zeros((0,2))
    for iter in range(1,n_iter+1):
        smiles = storage['iter' + str(iter)]['smis']
        scores_acq = np.concatenate((np.array([data[smile] for smile in smiles]), scores_acq), axis=0)
        hv = pg.hypervolume(scores_acq)
        hvs[iter-1] = hv.compute(ref)/hv_all.compute(ref)
    return hvs

def random_acq(n_iter,n_per_acq, n_repeats):
    data = read_data('data/selectivity_data.csv', 0, [1,2])

    xs = np.array(list(data.keys()))
    ys = np.array(list(data.values()))

    Y0 = ys[:,0]
    Y1 = ys[:,1]

    rng = np.random.default_rng()
    choices = np.array([rng.choice(len(xs), size=[n_iter,n_per_acq], replace=False) for i in range(n_repeats)])
    hvs = np.empty((n_repeats, n_iter))
    hv_all = pg.hypervolume(np.concatenate((np.expand_dims(Y0,1), np.expand_dims(Y1,1)), axis=1))

    for j in range(n_repeats):
        scores = np.zeros((n_iter,n_per_acq,2))
        x_scores = np.empty((n_iter,n_per_acq),dtype=object)
        ref = [np.max(Y0), np.max(Y1)]

        for i in range(n_iter):
            scores[i,:,0] = Y0[choices[j,i]]
            scores[i,:,1] = Y1[choices[j,i]]
            x_scores[i,:] = xs[choices[j,i]]
            pairs = np.concatenate((scores[0:(i+1),:,0].reshape(n_per_acq*(i+1),1), scores[0:(i+1),:,1].reshape(n_per_acq*(i+1),1)), axis=1)
            hv = pg.hypervolume(pairs)
            hvs[j,i] = hv.compute(ref)/hv_all.compute(ref)


    x_axis = np.arange(n_iter)
    hv_avs = np.mean(hvs,0)
    hv_stds = np.std(hvs,0)
    return hv_avs, hv_stds 

n_per_acq = 50
n_iter = 10
n_repeats = 1
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
kwargs.pop('init_size')
kwargs.pop('batch_sizes')

hv_store = np.zeros((n_repeats,n_iter))

for j in range(n_repeats):
    print('Starting Exploration ' + str(j) + '...')
    storage = {}
    storage['featurizer'] = featurizer.Featurizer(fingerprint=kwargs['fingerprint'],
                radius=kwargs['radius'], length=kwargs['length']
            )
    storage['pool'] = pools.pool(featurizer=storage['featurizer'],\
        **kwargs)
    storage['Acquirer'] = acquirer.Acquirer(size=len(storage['pool']),init_size=n_per_acq,batch_sizes=[n_per_acq],**kwargs)
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
    images = [plot_acquired(data, scores)]
    # plot_parity(storage['iter1'], storage['pool'], data, Y0_pred, 0), plot_parity(storage['iter1'], storage['pool'], data, Y1_pred, 1)
    acquired_scores = {}
    acquired_scores['1'] = new_scores

    for i in range(2,n_iter+1):
        iterlabel = 'iter' + str(i)
        new_scores = acquire_ndf(Y0_pred, Y1_pred, storage, iterlabel, scores)
        acquired_scores[str(i)] = new_scores
        clean_update_scores(scores, new_scores)
        retrain_models(storage, scores)
        Y0_pred, Y1_pred = predict(storage) 
        images.append(plot_acquired(data, scores))
        print(len(scores))

    hvs = calculate_hvs(storage,data)
    hv_store[j,:] = hvs

print(hv_store)
imageio.mimsave('./powers.gif', images)
x_axis = np.arange(n_iter)
hv_avs = np.mean(hv_store,axis=0)
hv_stds = np.std(hv_store,axis=0)

# random acquisition 
rand_avs, rand_std = random_acq(n_iter,n_per_acq, 5)

fig,ax = plt.subplots(1)
ax.plot(x_axis, hv_avs,label='Non-Dominated Sorting',color='C0')
ax.fill_between(x_axis, hv_avs-hv_stds, hv_avs+hv_stds, color='C0',alpha=0.2)
ax.plot(x_axis, rand_avs,label='Random',color='C1')
ax.fill_between(x_axis, rand_avs-rand_std, rand_avs+rand_std, color='C1',alpha=0.2)
ax.legend(loc='lower right')
ax.set_xlabel('Iteration')
ax.set_ylabel('Fraction of Total Hypervolume Acquired')
fig.show()
