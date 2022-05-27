import csv
import string
import numpy as np 
import heapq
from pathlib import Path
import pygmo as pg
import matplotlib.pyplot as plt

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



n_iter = 10
n_per_acq = 50
n_repeats = 3
def random_acq(n_iter,n_per_acq, n_repeats):
    data = read_data('data/selectivity_data.csv', 0, [1,2])

    xs = np.array(list(data.keys()))
    ys = np.array(list(data.values()))

    Y0 = -ys[:,0]
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

hv_avs,hv_stds = random_acq(5,50, 5)
plt.plot(x_axis, hv_avs,label='Random',color='C0')
plt.fill_between(x_axis, hv_avs-hv_stds, hv_avs+hv_stds, color='C0',alpha=0.2)
plt.show()
plt.legend(loc='lower right')
plt.xlabel('Iteration')
plt.ylabel('Fraction of Total Hypervolume Acquired')

print(Y0)