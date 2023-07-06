""" UMAP for dataset """

from read_results_utils import get_pool, top_k_smis_from_pool
from plot_utils import set_style, set_size
from pathlib import Path 
import umap.umap_ as umap
import random 
import math 
import numpy as np 
from matplotlib import pyplot as plt 
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
import pickle 
from tqdm import tqdm 

acq_func = 'pi'
c_type = 'objs'
base_dir = Path('results') / f'selective_IGF1R_clustering'
save_dir = Path(f'figure_scripts/molecular_diversity_figs')
fname = f'{save_dir}/top_15_fp_matrix.pkl'
obj_configs =  [Path(f'moo_runs/objective/IGF1R_min.ini'), Path(f'moo_runs/objective/CYP_max.ini')]
k = 0.15
if c_type=='None': 
    run_dir = Path('results') / 'selective_IGF1R_clustering' / f'seed-47-29_{acq_func}'
else:
    run_dir = Path('results') / 'selective_IGF1R_clustering' / f'seed-47-29_{acq_func}_{c_type}'

def sample_fps(pool, fraction):
    sample_size = math.ceil(len(pool)*fraction)
    fps = random.sample(list(pool.fps()), sample_size)
    return fps 

def simple_featurize(smi): 
    """ Same featurizer used in all studies for clustering """
    fp = rdmd.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), radius=2, nBits=2048, useChirality=True
    )

    X = np.empty(len(fp))
    ConvertToNumpyArray(fp, X)
    return X

def fit_reducer(fp_matrix, random_state=1, min_dist=0.01, n_neighbors=5):
    reducer = umap.UMAP(random_state=random_state, 
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        metric='jaccard',
    )
    reducer.fit(fp_matrix)

    transformed = reducer.transform(fp_matrix)
    return reducer, transformed

def plot_umap(background_t, main_t, fname='umap_top_15.png'):
    set_style()

    fig, ax = plt.subplots(1,1)
    ax.scatter(background_t[:,0], background_t[:,1], s=2, alpha=0.1, linewidth=0, color='lightgray')
    ax.scatter(main_t[:,0], main_t[:,1], s=2, alpha=1, linewidth=0, color='darkred', label=f'N={main_t.shape[0]}')
    ax.legend(loc='lower right',  handlelength=0)

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    set_size(1.5, 1.5, ax)
    fig.savefig(f'{save_dir}/{fname}')

def plot_basic_umap(transformed, fname='umap.png'):
    set_style()
    fig, ax = plt.subplots(1,1)
    ax.scatter(transformed[:,0], transformed[:,1], s=2, alpha=0.03, linewidth=0, color='gray')

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    set_size(1.5, 1.5, ax)
    fig.savefig(f'{save_dir}/{fname}')

def acquired_smis_from_folder(run_dir: Path): 
    iter_dirs = sorted((run_dir / 'chkpts').glob('iter_*'))
    smis = []
    for iter in iter_dirs: 
        with open(iter/'new_scores.pkl', 'rb') as file: 
            acquired = pickle.load(file)

        smis.append(list(acquired.keys()))
    
    return smis 

def acquired_top_smis(acquired_smis, top_smis): 
    return set(acquired_smis) & set(top_smis)

# pool = get_pool(base_dir, pool_sep='\t')
# top_smis = top_k_smis_from_pool(k, pool, obj_configs)
# top_smis = [str(smi) for smi in top_smis]

with open(Path(save_dir) / 'top_15_fp_matrix.pkl', 'rb') as f: 
    fp_matrix = pickle.load(f)

with open(Path(save_dir) / 'top_5_smis.pkl', 'rb') as f: 
    top_smis = pickle.load(f)

acquired_smis = acquired_smis_from_folder(run_dir)
top_acq = [list(acquired_top_smis(smis, top_smis)) for smis in acquired_smis]
top_fps = [ np.stack([simple_featurize(smi) for smi in smis]) for smis in top_acq]

reducer, transformed = fit_reducer(fp_matrix, random_state=30, min_dist=0.003, n_neighbors=8)

for i, fps in enumerate(top_fps):
    red_acq = reducer.transform(fps)
    plot_umap(transformed, red_acq, fname=f'iter{i}_{acq_func}_{c_type}.png') 



    