import numpy as np 
from pathlib import Path
import csv 
from molpal.models.nnmodels import (
        NNModel, NNDropoutModel, NNEnsembleModel, NNTwoOutputModel
    )
from molpal import args, Explorer, featurizer, pools, acquirer, models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygmo as pg
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import heapq

class LibraryDataset:
    def __init__(self, path, smiles_col, data_col) -> None:
        self.path = path
        self.smiles_col = smiles_col
        self.data_col = data_col 
        self.data = self.extract_data()
        self.smis = list(self.data.keys())

    def extract_data(self):
        reader = csv.reader(open(Path(self.path)))
        data = {}
        for row in reader: 
            try: 
                key = row[self.smiles_col]
                data[key]=float(row[self.data_col])
            except:
                pass
        return data

class Runner:
    def __init__(self, 
            fingerprint='Morgan',
            radius = 2,
            length = 2048,
            n_per_acq = 200, 
            num_objs = 2,
            num_models = 2,
        ) -> None:
        self.iteration = 0
        self.num_objs = num_objs
        self.length = length
        self.num_models = num_models
        
        self.featurizer = featurizer.Featurizer(
                fingerprint=fingerprint,
                radius=radius, 
                length=length
            )
        self.acquirer = acquirer.Acquirer(
                size=len(self.pool),
                init_size=n_per_acq,
                batch_sizes=[n_per_acq],
            )
        self.models = [self.init_model() for _ in range(num_models)]
        self.libraries = []
        self.scores = {}
        self.smis = []
    def init_libraries(self,data_list):
        self.libraries = [LibraryDataset(info[0], info[1], info[2]) \
            for info in data_list]
        self.all_smis = self.collect_smis()
        return self.libraries
    def collect_smis(self):
        all_smis = self.libraries[0].smis
        for i in range(1,len(self.libraries)):
            for smi in self.libraries[i].smis:
                if smi not in all_smis: 
                    all_smis.append(smi)
        return all_smis
    def init_model(self):
        model = NNEnsembleModel(
                    input_size=self.length, 
                    ensemble_size=3,
                    activation='relu'
                ) 
        return model 
    def run_iteration(self):
        if self.iteration == 0: 
            U = acquirer.metrics.random(np.empty(self.acquirer.size))
            heap = []
            for x, u in zip(self.smis, U):
                if len(heap) < self.acquirer.init_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))
            new_smis = [x for _, x in heap]
            self.scores = {}
            self.new_scores = {}
        for new in self.smis:
            self.new_scores[new] = [library.data[new] for library in self.libraries]
            else:
                self.new_scores = acquire_hvi(Y0_pred, Y1_pred, storage, iterlabel, scores, n_per_acq, data)

        clean_update_scores(self.scores, self.new_scores)

        self.iteration += 1



        

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

def get_metrics(Y0_pred, Y1_pred, data, scores):
    r2 = [r2_score(np.array(list(data.values()))[:,0], Y0_pred), 
            r2_score(np.array(list(data.values()))[:,1], Y1_pred)]
    mse = [mean_squared_error(np.array(list(data.values()))[:,0], Y0_pred),
            mean_squared_error(np.array(list(data.values()))[:,0], Y1_pred)]
    return np.array(mse), np.array(r2)


def plot_parity(scores, pool, data, Y_pred, n):
    smiles_list = list(pool.smis())
    Y_pred_dict = {smiles_list[i]: Y_pred[i] for i in range(len(smiles_list))}
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(np.array(list(data.values()))[:,n],Y_pred, color='gray',label='Not batched')
    ax.scatter([],[],color='blue', label='batched')
    for smiles in scores:
        ax.scatter(data[smiles][n],Y_pred_dict[smiles],color='blue')
    r2 = r2_score(np.array(list(data.values()))[:,n], Y_pred)
    mse = mean_squared_error(np.array(list(data.values()))[:,n], Y_pred)
    ax.plot([min(Y_pred),max(Y_pred)],[min(Y_pred),max(Y_pred)],color='k')
    ax.set_xlabel('True Docking Score')
    ax.set_ylabel('Predicted Docking Score')
    ax.legend(loc='upper left')
    # ax.legend('R2: '+str(r2), 'MSE: '+str(r2), loc='lower right')
    fig.savefig("parity.png")
    

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

def retrain_models(storage, scores, kwargs):
    scorelist = np.array(list(scores.items()),dtype=object)
    xs = scorelist[:,0]
    ys = np.vstack(scorelist[:,1])
    y0s = ys[:,0]
    y1s = ys[:,1]

    storage['model0'].train(xs, y0s,
                featurizer=storage['featurizer'], retrain = False)

    storage['model1'].train(xs, y1s,
                featurizer=storage['featurizer'], retrain = False)

def acquire_ndf(Y0_pred, Y1_pred, storage, iter_string, scores, n_per_acq, data):
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

def acquire_hvi(Y0_pred, Y1_pred, storage, iter_string, scores, n_per_acq, data):
    smiles_list = list(storage['pool'].smis())
    ref = np.max(np.array(list(data.values())),axis=0) + 2
    scores_acq = np.array(list(scores.values()))
    scores_pred = np.vstack((Y0_pred, Y1_pred)).transpose()
    hv_old = pg.hypervolume(scores_acq).compute(ref)
    hv_new = np.array([pg.hypervolume(np.vstack((scores_acq, scores_pred[i]))).compute(ref) 
                for i in range(len(scores_pred))])
    hvi = hv_new - hv_old
    sort_inds = np.argsort(-hvi)
    new_points = []
    i = 0
    while len(new_points) < n_per_acq:
        new_smile = smiles_list[sort_inds[i]]
        if new_smile in scores: 
            i = i + 1
        else:
            new_points.append(sort_inds[i]) 
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

def plot_boxplot(acquired_scores, data):
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

def calculate_hvs(storage,data, n_iter):
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

def plot_compare_hv(n_iter, n_per_acq, hv_store, fname, n_rand_rpt=5):
    rand_avs, rand_std = random_acq(n_iter,n_per_acq, n_rand_rpt)
    x_axis = np.arange(n_iter)
    hv_avs = np.mean(hv_store,axis=0)
    hv_stds = np.std(hv_store,axis=0)
    fig,ax = plt.subplots(1)
    ax.plot(x_axis, hv_avs,label='Non-Dominated Sorting',color='C0')
    ax.fill_between(x_axis, hv_avs-hv_stds, hv_avs+hv_stds, color='C0',alpha=0.2)
    ax.plot(x_axis, rand_avs,label='Random',color='C1')
    ax.fill_between(x_axis, rand_avs-rand_std, rand_avs+rand_std, color='C1',alpha=0.2)
    ax.legend(loc='lower right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fraction of Total Hypervolume Acquired')
    fig.show()
    plt.savefig(fname)