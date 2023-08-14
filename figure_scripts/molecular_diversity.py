""" UMAP for dataset """

from read_results_utils import get_pool, top_k_smis_from_pool
from plot_utils import set_style, set_size, cluster_labels, cluster_colors
from pathlib import Path 
# import umap.umap_ as umap
import umap
import random 
import math 
import numpy as np 
from matplotlib import pyplot as plt 
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
import pickle 
from tqdm import tqdm 
from sklearn.decomposition import PCA
import seaborn as sns

acq_func = 'pi'
c_type = 'objs'
base_dir = Path('results') / f'IGF1R_clustering'
save_dir = Path(f'figure_scripts/molecular_diversity_figs/')
obj_configs =  [Path(f'moo_runs/objective/IGF1R_min.ini'), Path(f'moo_runs/objective/CYP_max.ini')]
if c_type=='None': 
    run_dir = base_dir / f'seed-47-29_{acq_func}'
else:
    run_dir = base_dir / f'seed-47-29_{acq_func}_{c_type}'

def sample_fps(pool, fraction):
    sample_size = math.ceil(len(pool)*fraction)
    fps = random.sample(list(pool.fps()), sample_size)
    return fps 

def simple_featurize(smi): 
    """ Same featurizer used in all studies for clustering """
    try:
        fp = rdmd.GetHashedAtomPairFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), minLength=1, maxLength=1 + 2, nBits=2048
        )
        X = np.empty(len(fp))
        ConvertToNumpyArray(fp, X)
    except: 
        print(f'cannot featurize {smi}')
    return X

def fit_reducer(fp_matrix, random_state=1, min_dist=0.01, n_neighbors=5, n_pca=20):
    pca = PCA(n_components=n_pca, random_state=random_state)
    pca.fit(fp_matrix)

    pca_embeddings = pca.transform(fp_matrix)
    
    reducer = umap.UMAP(random_state=random_state, 
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='jaccard'
    )
    reducer.fit(pca_embeddings)

    transformed = reducer.transform(pca_embeddings)
    return pca, reducer, transformed

def fit_reducer_umap_only(fp_matrix, random_state=1, min_dist=0.01, n_neighbors=5):    
    reducer = umap.UMAP(random_state=random_state, 
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='jaccard'
    )
    reducer.fit(fp_matrix)

    transformed = reducer.transform(fp_matrix)
    return reducer, transformed

def plot_umap(background_t, main_t, fname='umap_top_15.png'):
    set_style()

    fig, ax = plt.subplots(1,1)
    sns.kdeplot(x=background_t[:,0], y=background_t[:,1], cmap=sns.color_palette("light:b", as_cmap=True), fill=True, bw_adjust=0.5)
    ax.scatter(main_t[:,0], main_t[:,1], s=0.7, alpha=0.3, linewidth=0, color='#880202', label=f'N={main_t.shape[0]}')
    ax.legend(loc='lower right',  handlelength=0)

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    set_size(1.5, 1.5, ax)
    fig.savefig(f'{save_dir}/{fname}')

def plot_basic_umap(transformed, fname='umap.png'):
    set_style()
    fig, ax = plt.subplots(1,1)
    ax.scatter(transformed[:,0], transformed[:,1], s=1, alpha=0.07, linewidth=0, color='gray')

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

def get_save_pool_info(k=None):
    pool = get_pool(base_dir, pool_sep='\t')

    if k is not None: 
        top_smis = top_k_smis_from_pool(k, pool, obj_configs)
        top_smis = [str(smi) for smi in top_smis]    
        fps = [simple_featurize(smi) for smi in top_smis]
        fp_matrix = np.stack(fps)
        reducer, transformed = fit_reducer_umap_only(fp_matrix, random_state=25, min_dist=0.01, n_neighbors=9)

        with open(Path(save_dir) / f'fp_matrix_top_{k}.pkl', 'wb') as f: 
            pickle.dump(fp_matrix, f)
        with open(Path(save_dir) / f'smis_top_{k}.pkl', 'wb') as f: 
            pickle.dump(top_smis, f)
        with open(Path(save_dir) / f'umap_reducer_top_{k}.pkl', 'wb') as f: 
            pickle.dump(reducer, f)
    
    else: 
        rand_fps = sample_fps(pool, fraction=1)
        fp_matrix = np.stack(rand_fps) 
        # reducer, transformed = fit_reducer_umap_only(fp_matrix, random_state=25, n_neighbors=10, min_dist=0.005)
        reducer, transformed = fit_reducer_umap_only(fp_matrix, random_state=25, n_neighbors=5, min_dist=0.01)

        plot_basic_umap(transformed=transformed)
        with open(Path(save_dir) / f'fp_matrix_random.pkl', 'wb') as f: 
            pickle.dump(fp_matrix, f)
        with open(Path(save_dir) / f'umap_reducer_random.pkl', 'wb') as f: 
            pickle.dump(reducer, f) 

    return  


def top_k_iterations(k):

    with open(Path(save_dir) / f'fp_matrix_top_{k}.pkl', 'rb') as f: 
        fp_matrix = pickle.load(f)
    with open(Path(save_dir) / f'smis_top_{k}.pkl', 'rb') as f: 
        top_smis = pickle.load(f)
    with open(Path(save_dir) / f'umap_reducer_top_{k}.pkl', 'rb') as f: 
        reducer = pickle.load(f)

    acquired_smis = acquired_smis_from_folder(run_dir)
    top_acq = [list(acquired_top_smis(smis, top_smis)) for smis in acquired_smis]
    top_fps = [ np.stack([simple_featurize(smi) for smi in smis]) for smis in top_acq]
        
    transformed = reducer.transform(fp_matrix)
    
    for i, fps in enumerate(top_fps):
        if i in {1, 3, 5}:
            red_acq = reducer.transform(fps)
            plot_umap(transformed, red_acq, fname=f'iter{i}_{acq_func}_{c_type}.pdf') 

def empty_kde(k=None):
    if k: 
        with open(Path(save_dir) / f'fp_matrix_top_{k}.pkl', 'rb') as f: 
            fp_matrix = pickle.load(f)
        with open(Path(save_dir) / f'umap_reducer_top_{k}.pkl', 'rb') as f: 
            reducer = pickle.load(f)
    else: 
        with open(Path(save_dir) / f'fp_matrix_random.pkl', 'rb') as f: 
            fp_matrix = pickle.load(f)
        with open(Path(save_dir) / f'umap_reducer_random.pkl', 'rb') as f: 
            reducer = pickle.load(f)

    transformed = reducer.transform(fp_matrix)
    
    set_style()

    fig, ax = plt.subplots(1,1)
    sns.kdeplot(x=transformed[:,0], y=transformed[:,1], cmap=sns.color_palette("light:b", as_cmap=True), fill=True, bw_adjust=0.5)
    ax.scatter([],[])

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    set_size(1.5, 1.5, ax)
    fig.savefig(f'{save_dir}/empty_kde_{k}.pdf', bbox_inches='tight', dpi=200, transparent=True)


def random_iterations():

    with open(Path(save_dir) / f'fp_matrix_random.pkl', 'rb') as f: 
        fp_matrix = pickle.load(f)
    with open(Path(save_dir) / f'umap_reducer_random.pkl', 'rb') as f: 
        reducer = pickle.load(f)

    acquired_smis = acquired_smis_from_folder(run_dir)
    acq_fps = [ np.stack([simple_featurize(smi) for smi in smis]) for smis in acquired_smis]
        
    transformed = reducer.transform(fp_matrix)
    
    for i, fps in enumerate(acq_fps):
        if i in {1, 3, 5}:
            red_acq = reducer.transform(fps)
            plot_umap(transformed, red_acq, fname=f'random/iter{i}_{acq_func}_{c_type}.pdf') 


def random_15_umap(): 
    # pool = get_pool(base_dir, pool_sep='\t')
    # fps = sample_fps(pool, 0.1)
    # fp_matrix = np.stack(fps) 
    with open(Path(save_dir)/ 'fp_matrix_random.pkl', 'wb') as f: 
        pickle.dump(fp_matrix, f)

    with open(Path(save_dir)/ 'fp_matrix_random.pkl', 'rb') as f: 
        fp_matrix = pickle.load(f)

    pca, reducer, transformed = fit_reducer(fp_matrix, random_state=30, min_dist=0.000001, n_neighbors=6, n_pca=40)
    # pca, reducer, transformed = fit_reducer(fp_matrix, random_state=30, min_dist=0.00001, n_neighbors=7, n_pca=40)

    acquired_smis = acquired_smis_from_folder(run_dir)
    all_smis = [item for sublist in acquired_smis for item in sublist]
    acquired_fps = np.stack([simple_featurize(smi) for smi in all_smis])

    pca_embeddings = pca.transform(acquired_fps)
    red_acq = reducer.transform(pca_embeddings)
    plot_umap(transformed, red_acq, fname=f'all_random_{acq_func}_{c_type}_debug.pdf') 

if __name__=='__main__': 
    # get_save_pool_info()
    # get_save_pool_info(k=0.1)
    # random_15_umap()
    # top_k_iterations(0.05)
    random_iterations()
    # empty_kde()
    # print('done')



    