import datetime
import os
import signal
import sys
import ray 
from molpal import args, Explorer, featurizer, pools, acquirer
import numpy as np 
import heapq

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

# first iteration
print('Starting Exploration ...')
storage = {}
storage['featurizer'] = featurizer.Featurizer(fingerprint=kwargs['fingerprint'],
            radius=kwargs['radius'], length=kwargs['length']
        )
storage['pool'] = pools.pool(featurizer=storage['featurizer'],\
    **kwargs)
storage['Acquirer'] = acquirer.Acquirer(size=len(storage['pool']),**kwargs)
xs = storage['pool'].smis()

U = acquirer.metrics.random(np.empty(storage['Acquirer'].size))

heap = []
for x, u in zip(xs, U):
    if len(heap) < storage['Acquirer'].init_size:
        heapq.heappush(heap, (u, x))
    else:
        heapq.heappushpop(heap, (u, x))

