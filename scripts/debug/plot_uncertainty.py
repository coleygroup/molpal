""" To look at quality of uncertainty quantification """

from pathlib import Path 
import numpy as np
from molpal import args, pools, featurizer
from molpal.acquirer.pareto import Pareto
from molpal.objectives.lookup import LookupObjective
from configargparse import ArgumentParser 
from typing import List, Union, Dict
from tqdm import tqdm
import csv 
from scipy import stats
import matplotlib.pyplot as plt
from plot_utils import set_size, set_style

def get_pool(config_file, pool_sep = None): 
    parser = ArgumentParser()
    args.add_pool_args(parser)
    args.add_general_args(parser)
    args.add_encoder_args(parser)
    params, _ = parser.parse_known_args("--config " + str(config_file))
    params = vars(params)
    if pool_sep: 
        params['delimiter'] = pool_sep

    feat = featurizer.Featurizer(
        fingerprint=params['fingerprint'],
        radius=params['radius'], length=params['length']
    )
    pool = pools.pool(featurizer=feat, **params)
    return pool

def extract_preds(expt):
    preds = {}
    for folder in (expt / 'chkpts').glob('*/'):
        if folder.stem == 'iter_1':
            continue 
        preds_npz = np.load(folder/'preds.npz')
        preds[folder.stem] = {}
        preds[folder.stem]['Y_pred'] = preds_npz['Y_pred']
        preds[folder.stem]['Y_var'] = preds_npz['Y_var']
    
    return preds

def build_true_dict(true_csv, smiles_col: int, score_cols: int = 1,
                    title_line: bool = True,
                    maximize: List[bool] = [False, False], 
                    delimiter: str = ',') -> Dict[str, float]:
    
    c = [1 if maxx else -1 for maxx in maximize]

    with open(true_csv) as fid:
        reader = csv.reader(fid, delimiter = delimiter)
        if title_line:
            next(reader)

        d_smi_score = {}
        for row in tqdm(reader, desc='Building true dict', leave=False):
            try:
                d_smi_score[row[smiles_col]] = [cc * float(row[score_col]) for cc, score_col in zip(c, score_cols) ]
            except ValueError:
                continue

    return d_smi_score

def extract_Y_true(d_smi_score, pool, preds):
    Y_true = np.empty((len(pool),preds['iter_2']['Y_pred'].shape[1]))
    for i, smi in enumerate(pool.smis()): 
        Y_true[i,:] = np.array(d_smi_score[smi])
    
    return Y_true 

def calc_residuals(Y_true, preds): 
    residuals = {}
    for iter, pred in preds.items(): 
        residuals[iter] = pred['Y_pred'] - Y_true
    
    return residuals
    
def calibration_curve(resid: np.array, Y_var: np.array): 
    norm = stats.norm(loc=0, scale=1)    
    stdevs = resid + np.random.normal(loc=0.0, scale=1.0, size=resid.shape) # np.sqrt(Y_var)
    normalized_residuals = resid / stdevs
    predicted_pi = np.linspace(0, 1, 100)
    observed_pi = []
    for percentile in predicted_pi:
        upper_bound = norm.ppf(percentile)
        # Count how many residuals fall inside here
        num_within_quantile = (normalized_residuals <= upper_bound).sum()
        observed_pi.append(num_within_quantile / len(resid))
    return predicted_pi, np.array(observed_pi)

def calibrate_all(residuals: Dict[str,np.array], preds: Dict[str, Dict]):
    calibration_curves = {}
    for iter, pred in tqdm(preds.items(), desc='Calibration Curves'):
        calibration_curves[iter] = []
        for i in range(0, pred['Y_pred'].shape[1]):
            pred_pi, obs_pi = calibration_curve(
                                    residuals[iter][:,i],
                                    Y_var = pred['Y_var'][:,i],
                                )   
            calibration_curves[iter].append({
                'predicted_pi': pred_pi,
                'observed_pi': obs_pi,
            })
    return calibration_curves

def plot_cal_curves(calibration_curves, filename):
    """ plots calibration curves. assumes two models """
    fig, axs = plt.subplots(1, 2)
    for iter, curves in calibration_curves.items(): 
        for i, curve in enumerate(curves):
            axs[i].plot(curve['predicted_pi'], curve['observed_pi'], label=f"Iter {iter.split('_')[1]}") 
    for i, ax in enumerate(axs): 
        handles, labels = ax.get_legend_handles_labels()
        order = np.argsort(labels)
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right') 
        ax.plot([0,1], [0,1], '--k')
        ax.set_title(f"Model {i+1} Calibration")
    set_size(7,3)    
    set_style()
    plt.savefig(filename)
    return

def plot_sharpness(preds, filename):
    fig, ax = plt.subplots()
    # ax.bar( ,label='Model 1')

    # ax.bar( ,label='Model 2')


    set_size(3,3)    
    set_style()
    plt.savefig(filename)
    return
        

expt = Path('results/moo_results_2023-06-01-16-11_recalibrate/seed-47-29_ei') # Path('results/moo_results_2023-06-01-23-dockstring-nn')
library = Path('data/dockstring-dataset.tsv')
pool = get_pool(expt / "config.ini", pool_sep='\t')
preds = extract_preds(expt)
d_smi_score = build_true_dict(library, 
                              smiles_col=1, 
                              score_cols=[32, 33],
                              title_line=True, 
                              maximize = [False, True],
                              delimiter='\t',
                              )
Y_true = extract_Y_true(d_smi_score, pool, preds)
residuals = calc_residuals(Y_true, preds)
calibration_curves = calibrate_all(residuals, preds)
plot_cal_curves(calibration_curves, expt / 'calibration_curve.png')