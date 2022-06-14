import datetime
import os
import signal
import sys
import ray 
from molpal import args, Explorer, featurizer, pools, acquirer, models
import numpy as np 
import heapq
import matplotlib.pyplot as plt
import imageio
from molpal.models.nnmodels import (
        NNModel, NNDropoutModel, NNEnsembleModel, NNTwoOutputModel
    )

from moo_utils import (
    acquire_hvi, clean_update_scores, read_data, plot_parity, plot_acquired, 
    retrain_models, acquire_ndf, predict, plot_boxplot, calculate_hvs,
    random_acq, plot_compare_hv
)



# To start: in vscode, check interpreter in conda environment 'molpalmoo'
# kill current termial 
# go to menu bar > terminal > new terminal 
# type in ' ray start --head  ' (include --num-gpus N --num-cpus N if desired)

# settings 
n_per_acq = 200
n_iter = 5
n_repeats = 1
data = read_data('data/selectivity_data.csv', 0, [1,2])

# shutdown any previous ray cluster 
if ray.is_initialized(): ray.shutdown()

# set up ray cluster 
ray.init(num_cpus=5, num_gpus=1)
print(ray.cluster_resources())

# setup run
arguments = '-o lookup -l data/selectivity_data.csv --output-dir selectivity -v \
--objective-config examples/objective/selectivity.ini'.split()
params = vars(args.gen_args(arguments))
params['models'] = ['rf', 'rf']
params['metric'] = 'greedy'
path = params.pop("output_dir")
kwargs = params
kwargs.pop('init_size')
kwargs.pop('batch_sizes')

hv_nds_store = np.zeros((n_repeats,n_iter))
hv_hvi_store = np.zeros((n_repeats,n_iter))


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
    storage['model0'] = NNModel(input_size=kwargs['length'], num_tasks=1, \
                    test_batch_size=3000,  # so one batch wouldn't have like 10 molecules 
                    layer_sizes=None, 
                    dropout=None,
                    activation='relu',
                    uncertainty=None,
                    model_seed=None) 
    storage['model1'] = NNModel(input_size=kwargs['length'], num_tasks=1, \
                    test_batch_size=3000,  
                    layer_sizes=None, 
                    dropout=None,
                    activation='relu',
                    uncertainty=None,
                    model_seed=None) 

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

    retrain_models(storage, scores, kwargs)

    Y0_pred, Y1_pred = predict(storage)
    plot_parity(storage['iter1'], storage['pool'], data, Y0_pred, 0)
    # images = [plot_acquired(data, scores)]
    # plot_parity(storage['iter1'], storage['pool'], data, Y0_pred, 0), plot_parity(storage['iter1'], storage['pool'], data, Y1_pred, 1)
    acquired_scores = {}
    acquired_scores['1'] = new_scores

    for i in range(2,n_iter+1):
        iterlabel = 'iter' + str(i)
        new_scores = acquire_hvi(Y0_pred, Y1_pred, storage, iterlabel, scores, n_per_acq, data)
        acquired_scores[str(i)] = new_scores
        clean_update_scores(scores, new_scores)
        retrain_models(storage, scores, kwargs)
        Y0_pred, Y1_pred = predict(storage) 
        plot_parity(storage[iterlabel], storage['pool'], data, Y0_pred, 0)
        # images.append(plot_acquired(data, scores))
        print(len(scores))

    hvs = calculate_hvs(storage,data,n_iter)
    hv_nds_store[j,:] = hvs

print(hv_nds_store)
# imageio.mimsave('./powers.gif', images)


# random acquisition 
fname = 'compare_hv.png'
plot_compare_hv(n_iter, n_per_acq, hv_nds_store, fname, n_rand_rpt=5)
